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
Acceptance criteria for gpu4pyscf.pbc.gto.ecp.ecp_int (PBC scalar ECP).

Compares against pyscf.pbc.gto.ecp.ecp_int at Gamma and on a k-point mesh.
Skips (xfails) until ecp_int is implemented -- see docs/ecp-pbc-design.md.
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
        self.assertLess(abs(np.asarray(ref) - got).max(), 1e-9)

    def test_gamma_is_real(self):
        got = cp.asnumpy(gpu_ecp.ecp_int(self.cell))
        self.assertLess(abs(got.imag).max() if np.iscomplexobj(got) else 0.0,
                        1e-12)

    def test_kpts(self):
        kpts = self.cell.make_kpts([2, 1, 2])
        ref = cpu_ecp.ecp_int(self.cell, kpts)              # [nkpts, nao, nao]
        got = cp.asnumpy(gpu_ecp.ecp_int(self.cell, kpts))
        self.assertEqual(got.shape, np.asarray(ref).shape)
        self.assertLess(abs(np.asarray(ref) - got).max(), 1e-9)

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


if __name__ == '__main__':
    print('PBC ECP acceptance tests')
    unittest.main()
