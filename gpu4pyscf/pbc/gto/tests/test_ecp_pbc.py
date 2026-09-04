# Copyright 2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Acceptance criteria for gpu4pyscf.pbc.gto.ecp.ecp_int -- PBC scalar ECP
(intor='ECPscalar') and spin-orbit ECP (intor='ECPso').

Compares against pyscf.pbc.gto.ecp.ecp_int at Gamma and on a k-point mesh;
checks Hermiticity, rcut convergence, ket-image batch invariance, and that
pbc/scf get_hcore picks up the ECP.  See docs/ecp-pbc-design.md.
'''

import unittest
import numpy as np

try:
    import cupy as cp
    from pyscf.pbc import gto as pbcgto
    from pyscf.pbc.gto import ecp as cpu_ecp
    from gpu4pyscf.pbc.gto import ecp as gpu_ecp
    _IMPORTED = True
except Exception:                       # noqa: BLE001  (env may lack cupy/pbc)
    _IMPORTED = False


def _cell():
    # 1-D chain of I atoms with a def2 ECP; small, heavy-atom, ECP-bearing.
    cell = pbcgto.Cell()
    cell.atom = 'I 0 0 0; I 0 0 3.0'
    cell.a = np.eye(3) * 6.0
    cell.a[2, 2] = 6.0
    cell.basis = 'def2-svp'
    cell.ecp = 'def2-svp'
    cell.dimension = 3
    cell.precision = 1e-10
    cell.verbose = 0
    cell.build()
    return cell


def _implemented():
    if not _IMPORTED:
        return False
    try:
        gpu_ecp.ecp_int(_cell())
        return True
    except NotImplementedError:
        return False
    except Exception:                   # noqa: BLE001
        return True                     # implemented but erroring -- let it show


@unittest.skipUnless(_IMPORTED, 'cupy / pyscf.pbc not importable')
@unittest.skipUnless(_implemented(), 'gpu4pyscf.pbc.gto.ecp.ecp_int not implemented')
class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = _cell()

    def test_gamma(self):
        ref = cpu_ecp.ecp_int(self.cell)                    # [nao, nao] real
        got = cp.asnumpy(gpu_ecp.ecp_int(self.cell))
        self.assertEqual(got.shape, ref.shape)
        # 5e-9: GPU uses 128-pt Gauss-Chebyshev radial quadrature; CPU uses a
        # different scheme.  Both are converged; the gap is not a bug.
        self.assertLess(abs(np.asarray(ref) - got).max(), 5e-9)

    def test_gamma_is_real(self):
        got = cp.asnumpy(gpu_ecp.ecp_int(self.cell))
        self.assertLess(abs(got.imag).max() if np.iscomplexobj(got) else 0.0,
                        1e-12)

    def test_kpts(self):
        kpts = self.cell.make_kpts([2, 1, 2])
        ref = cpu_ecp.ecp_int(self.cell, kpts)              # [nkpts, nao, nao]
        got = cp.asnumpy(gpu_ecp.ecp_int(self.cell, kpts))
        self.assertEqual(got.shape, np.asarray(ref).shape)
        self.assertLess(abs(np.asarray(ref) - got).max(), 5e-9)

    def test_kpt_hermitian(self):
        kpts = self.cell.make_kpts([2, 2, 1])
        got = cp.asnumpy(gpu_ecp.ecp_int(self.cell, kpts))
        for k in range(len(kpts)):
            self.assertLess(abs(got[k] - got[k].conj().T).max(), 1e-10)

    def test_rcut_converged(self):
        # result stable when the lattice-sum cutoff is enlarged
        a = cp.asnumpy(gpu_ecp.ecp_int(self.cell))
        c2 = _cell()
        c2.rcut = self.cell.rcut * 1.5
        b = cp.asnumpy(gpu_ecp.ecp_int(c2))
        self.assertLess(abs(a - b).max(), 1e-9)

    def test_image_batching_invariant(self):
        # phase-4 ket-image batching must not change the result
        import gpu4pyscf.pbc.gto.ecp as m
        saved = m._image_batch_size
        try:
            m._image_batch_size = lambda *a, **k: 1        # one image per batch
            got1 = cp.asnumpy(gpu_ecp.ecp_int(self.cell))
            m._image_batch_size = lambda *a, **k: 10**9     # all in one batch
            gotN = cp.asnumpy(gpu_ecp.ecp_int(self.cell))
        finally:
            m._image_batch_size = saved
        self.assertLess(abs(got1 - gotN).max(), 1e-12)


def _so_cell():
    '''Small periodic cell with a spin-orbit ECP, or None if the SO set does
    not cover a convenient element in this PySCF build.'''
    for elem, ecp_name, a in (('I', 'ecpds28mdfso', 7.0),
                              ('Cs', 'ecpds46mdfso', 8.0),
                              ('Xe', 'ecpds28mdfso', 7.0)):
        try:
            cell = pbcgto.Cell()
            cell.atom = f'{elem} 0 0 0'
            cell.a = np.eye(3) * a
            cell.basis = 'def2-svp'
            cell.ecp = {elem: ecp_name}
            cell.precision = 1e-9
            cell.verbose = 0
            cell.build()
        except Exception:                       # noqa: BLE001
            continue
        if len(cell._ecpbas) and np.any(cell._ecpbas[:, 4] == 1):  # SO_TYPE_OF
            return cell
    return None


@unittest.skipUnless(_IMPORTED, 'cupy / pyscf.pbc not importable')
@unittest.skipUnless(_implemented(), 'gpu4pyscf.pbc.gto.ecp.ecp_int not implemented')
class SOKnownValues(unittest.TestCase):
    '''PBC spin-orbit ECP vs pyscf.pbc.gto.ecp.ecp_int(..., intor='ECPso').'''

    @classmethod
    def setUpClass(cls):
        cls.cell = _so_cell()

    def setUp(self):
        if self.cell is None:
            self.skipTest('no periodic SO-ECP element available in this build')

    def test_gamma_so(self):
        ref = cpu_ecp.ecp_int(self.cell, intor='ECPso')      # [2nao, 2nao]
        got = cp.asnumpy(gpu_ecp.ecp_int(self.cell, intor='ECPso'))
        self.assertEqual(got.shape, np.asarray(ref).shape)
        self.assertLess(abs(np.asarray(ref) - got).max(), 5e-9)

    def test_kpts_so(self):
        kpts = self.cell.make_kpts([2, 1, 1])
        ref = cpu_ecp.ecp_int(self.cell, kpts, intor='ECPso')
        got = cp.asnumpy(gpu_ecp.ecp_int(self.cell, kpts, intor='ECPso'))
        self.assertEqual(got.shape, np.asarray(ref).shape)
        self.assertLess(abs(np.asarray(ref) - got).max(), 5e-9)

    def test_so_hermitian(self):
        kpts = self.cell.make_kpts([2, 2, 1])
        got = cp.asnumpy(gpu_ecp.ecp_int(self.cell, kpts, intor='ECPso'))
        for k in range(len(kpts)):
            self.assertLess(abs(got[k] - got[k].conj().T).max(), 1e-10)


@unittest.skipUnless(_IMPORTED, 'cupy / pyscf.pbc not importable')
@unittest.skipUnless(_implemented(), 'gpu4pyscf.pbc.gto.ecp.ecp_int not implemented')
class SCFWiring(unittest.TestCase):
    '''Verify that get_hcore includes the ECP contribution.'''

    def test_get_hcore_includes_ecp(self):
        from gpu4pyscf.pbc.scf import rhf as gpu_rhf
        cell = pbcgto.Cell()
        cell.atom = 'I 0 0 0'
        cell.a = np.eye(3) * 8.0
        cell.basis = 'def2-svp'
        cell.ecp = 'def2-svp'
        cell.precision = 1e-6
        cell.mesh = [13, 13, 13]
        cell.verbose = 0
        cell.build()
        self.assertTrue(len(cell._ecpbas) > 0, 'cell has no ECP basis')

        mf = gpu_rhf.RHF(cell)
        hcore = cp.asnumpy(mf.get_hcore(cell))
        self.assertTrue(np.isfinite(hcore).all(), 'hcore contains NaN/Inf')

        # ECP contribution is nonzero; subtracting it changes the matrix
        h_ecp = cp.asnumpy(gpu_ecp.ecp_int(cell))
        self.assertGreater(abs(h_ecp).max(), 1.0, 'ECP matrix is unexpectedly small')
        diff = abs(abs(hcore).max() - abs(hcore - h_ecp).max())
        self.assertGreater(diff, 0.01, 'get_hcore does not include ECP')


if __name__ == '__main__':
    print('PBC ECP acceptance tests')
    unittest.main()
