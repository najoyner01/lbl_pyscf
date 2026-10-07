import warnings, time, sys
warnings.simplefilter('always')
import cupy
print('cupy', cupy.__version__, 'runtime', cupy.cuda.runtime.runtimeGetVersion())
from gpu4pyscf.lib import cutensor as ct
print('gpu4pyscf cutensor available:', ct.cutensor is not None, 'engine:', ct.contract_engine)
import gpu4pyscf, pyscf
print('gpu4pyscf', gpu4pyscf.__file__)
from gpu4pyscf.lib.cupy_helper import contract
a = cupy.random.rand(40,50,60); b = cupy.random.rand(60,50)
r = contract('ijk,kj->ij', a, b); ref = cupy.einsum('ijk,kj->ij', a, b)
print('contract vs einsum max diff', float(abs(r-ref).max()))
from pyscf import gto
from gpu4pyscf import dft, scf
mol = gto.M(atom='Ag 0 0 0; Cl 0 0 2.28', basis='def2-tzvp', ecp='def2-tzvp', verbose=0)
for xc in ('pbe','b3lyp'):
    mf = dft.RKS(mol, xc=xc); mf.grids.atom_grid=(75,302); mf.conv_tol=1e-9
    t=time.time(); e=mf.kernel(); t=time.time()-t
    cut = getattr(mf._opt_gpu.get(None), '_diffuse_cutoff_used', 'n/a') if mf._opt_gpu.get(None) is not None else 'no vhfopt'
    print(f'{xc:6s} E={e:.10f} {t:6.2f}s cycles={mf.cycles} diffuse_cutoff_used={cut}')
    mfc = mf.to_cpu(); ec = mfc.kernel()
    print(f'       CPU E={ec:.10f} |dE|={abs(e-ec):.2e}')
