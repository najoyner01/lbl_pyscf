"""Dedicated-GPU component profile for the TM-ladder hot spots. Read-only diagnostics."""
import time, io, re, collections, sys
import cupy, numpy as np
from pyscf import gto, lib
from gpu4pyscf.dft import rks
from gpu4pyscf.scf import jk, j_engine
from gpu4pyscf.pbc.gto.int1e import int1e_kin
from gpu4pyscf.df.int3c2e_bdiv import Int3c2eOpt, contract_int3c2e_auxvec
import gpu4pyscf.gto.ecp as gecp
from gpu4pyscf.lib import logger

def sync(): cupy.cuda.Stream.null.synchronize()
def T(f, n=3):
    sync(); f(); sync(); t0=time.perf_counter()
    for _ in range(n): f()
    sync(); return (time.perf_counter()-t0)/n

SYS = [('AgCl','Ag 0 0 0; Cl 0 0 2.28',0),
       ('MoF6','Mo 0 0 0; F 1.82 0 0; F -1.82 0 0; F 0 1.82 0; F 0 -1.82 0; F 0 0 1.82; F 0 0 -1.82',0),
       ('IrCl6','Ir 0 0 0; Cl 2.35 0 0; Cl -2.35 0 0; Cl 0 2.35 0; Cl 0 -2.35 0; Cl 0 0 2.35; Cl 0 0 -2.35',-3)]
BASES = ['def2-svp','def2-tzvp','def2-qzvp']

# warm up kernels / libxc
m=gto.M(atom='Ag 0 0 0; Cl 0 0 2.28',basis='def2-qzvp',ecp='def2-qzvp',verbose=0)
mf=rks.RKS(m,xc='b3lyp'); mf.max_cycle=1; mf.kernel()
for xc in ['pbe']:
    mf=rks.RKS(m,xc=xc); mf.max_cycle=1; mf.kernel()

for name,atom,chg in SYS:
    for basis in BASES:
        mol=gto.M(atom=atom,basis=basis,ecp=basis,charge=chg,verbose=0)
        print(f'\n=== {name} {basis} nao={mol.nao} nbas={mol.nbas} natm={mol.natm}', flush=True)
        nucmol = gto.mole.fakemol_for_charges(mol.atom_coords())
        Z = cupy.asarray(mol.atom_charges(), dtype=np.float64)
        print(f'  Int3c2eOpt.build (nuc)     {T(lambda: Int3c2eOpt(mol,nucmol).build())*1000:8.1f} ms')
        print(f'  int1e_nuc total            {T(lambda: contract_int3c2e_auxvec(mol,nucmol,-Z))*1000:8.1f} ms')
        print(f'  int1e_kin                  {T(lambda: int1e_kin(mol))*1000:8.1f} ms')
        print(f'  get_ecp                    {T(lambda: gecp.get_ecp(mol))*1000:8.1f} ms')
        t0=time.perf_counter(); mol.intor('ECPscalar'); print(f'  CPU ECPscalar              {(time.perf_counter()-t0)*1000:8.1f} ms')
        print(f'  jk._VHFOpt.build           {T(lambda: jk._VHFOpt(mol).build())*1000:8.1f} ms')
        print(f'  j_engine._VHFOpt.build     {T(lambda: j_engine._VHFOpt(mol).build())*1000:8.1f} ms')
        mf=rks.RKS(mol,xc='b3lyp'); mf.grids.atom_grid=(75,302)
        print(f'  grids.build                {T(lambda: mf.grids.build())*1000:8.1f} ms')
        print(f'  get_init_guess             {T(lambda: mf.get_init_guess(mol))*1000:8.1f} ms')
        dm=mf.get_init_guess(mol); mf.grids.build()
        vhfopt=jk._VHFOpt(mol).build(); jopt=j_engine._VHFOpt(mol).build()
        print(f'  j_engine.get_j             {T(lambda: j_engine.get_j(mol,dm,1,jopt))*1000:8.1f} ms')
        print(f'  jk.get_jk (J+K)            {T(lambda: jk.get_jk(mol,dm,hermi=1,vhfopt=vhfopt))*1000:8.1f} ms')
        print(f'  jk.get_k                   {T(lambda: jk.get_jk(mol,dm,hermi=1,vhfopt=vhfopt,with_j=False))*1000:8.1f} ms')
        ni=mf._numint
        print(f'  nr_rks vxc (b3lyp)         {T(lambda: ni.nr_rks(mol,mf.grids,"b3lyp",dm))*1000:8.1f} ms   ngrids={mf.grids.coords.shape[0]}')
        # per l-quartet breakdown of get_jk
        mol2=mol.copy(); mol2.verbose=logger.DEBUG1; mol2.stdout=io.StringIO()
        v2=jk._VHFOpt(mol2).build(); jk.get_jk(mol2,dm,hermi=1,vhfopt=v2)
        mol2.stdout=io.StringIO(); sync(); jk.get_jk(mol2,dm,hermi=1,vhfopt=v2); sync()
        tot=collections.Counter(); tasks={}
        for l in mol2.stdout.getvalue().splitlines():
            mm=re.search(r'processing (\(\w\w\|\w\w\)).*tasks ~= (\d+).*wall time\s+([\d.]+)',l)
            if mm: tot[mm.group(1)]+=float(mm.group(3)); tasks[mm.group(1)]=int(mm.group(2))
        s=sum(tot.values()) or 1e-9
        print(f'  get_jk per-group sum={s:.3f}s launches={len(tot)}; by max-l of quartet:')
        byl=collections.Counter()
        for k,v in tot.items():
            lmax=max('spdfghi'.index(c) for c in k if c.isalpha()); byl[lmax]+=v
        print('     ', {f'lmax={k}':f'{v:.3f}s' for k,v in sorted(byl.items())})
        for k,v in tot.most_common(6): print(f'      {k} {v:7.3f}s {100*v/s:5.1f}% tasks~{tasks[k]}')
        # EXPERIMENT: K build with full primitive decontraction (what j_engine already does)
        from gpu4pyscf.gto import mole as gmole
        orig=gmole.SortedGTO.from_mol
        def patched(m, decontract=True, diffuse_cutoff=0.3, **kw):
            return orig(m, decontract=decontract, diffuse_cutoff=1e200, **kw)
        gmole.SortedGTO.from_mol=staticmethod(patched) if isinstance(gmole.SortedGTO.__dict__.get('from_mol'),staticmethod) else classmethod(lambda cls,m,**kw: orig(m,**{**kw,'diffuse_cutoff':1e200}))
        try:
            pv=jk._VHFOpt(mol).build()
            print(f'  [exp] decontracted: groups={len(pv.sorted_mol.uniq_l_ctr)} nbas={pv.sorted_mol.nbas}; VHFOpt.build {T(lambda: jk._VHFOpt(mol).build())*1000:8.1f} ms')
            vk_ref=jk.get_jk(mol,dm,hermi=1,vhfopt=vhfopt,with_j=False)[1]
            vk_new=jk.get_jk(mol,dm,hermi=1,vhfopt=pv,with_j=False)[1]
            print(f'  [exp] decontracted get_k    {T(lambda: jk.get_jk(mol,dm,hermi=1,vhfopt=pv,with_j=False))*1000:8.1f} ms   max|dK|={float(abs(vk_ref-vk_new).max()):.2e}')
            for xc in ['b3lyp']:
                mf2=rks.RKS(mol,xc=xc); mf2.grids.atom_grid=(75,302); mf2.conv_tol=1e-9; mf2.verbose=0
                sync(); t0=time.perf_counter(); e2=mf2.kernel(); sync()
                print(f'  [exp] decontracted SCF {xc} wall={time.perf_counter()-t0:7.2f}s cycles={mf2.cycles} E={e2:.10f}')
        finally:
            gmole.SortedGTO.from_mol=orig
        # full SCF split: init vs cycles
        for xc in ['pbe','b3lyp']:
            mf=rks.RKS(mol,xc=xc); mf.grids.atom_grid=(75,302); mf.conv_tol=1e-9; mf.verbose=5; mf.stdout=io.StringIO()
            sync(); t0=time.perf_counter(); mf.kernel(); sync(); wall=time.perf_counter()-t0
            out=mf.stdout.getvalue()
            init=re.search(r'SCF initialization.*?wall time\s+([\d.]+)',out)
            cyc=[float(x) for x in re.findall(r'for cycle=\s*\d+.*wall time\s+([\d.]+)',out)]
            print(f'  SCF {xc:6s} wall={wall:7.2f}s cycles={mf.cycles} init={init.group(1) if init else "?"}s  per-cycle mean={np.mean(cyc) if cyc else 0:.3f}s')
        del mf, vhfopt, jopt; cupy.get_default_memory_pool().free_all_blocks()
