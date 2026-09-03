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
ECP-atom slicing / batching in gpu4pyscf.gto.ecp: iterating the first- and
second-derivative integrals over ECP atoms in chunks must give bit-for-bit the
same result as the single-shot dense build, and the reduction helpers
(get_ecp_ip_sum / get_ecp_ipip_sum) must equal the summed dense tensor.
'''

import unittest
import numpy as np
import cupy as cp
from pyscf import gto

from gpu4pyscf.gto import ecp


def setUpModule():
    global mol
    ecp_na = gto.basis.parse_ecp('''
Na nelec 10
Na ul
2       1.0                   0.5
Na S
2      13.652203             732.2692
2       6.826101              26.484721
Na P
2      10.279868             299.489474
2       5.139934              26.466234
Na D
2       7.349859             124.457595
2       3.674929              14.035995
''')
    basis_na = gto.basis.parse('''
     Na    S
           2.8000000              0.0210870             -0.0045400
           1.3190000              0.3461290             -0.1703520
           0.9059000              0.0393780              0.1403820
     Na    P
           1.2000000              0.0000000              0.5000000
           0.3827000              0.5010080              1.0000000
     Na    D
           0.3827000              1.0000000
''')
    # four ECP-bearing atoms so batch_size in {1,2,3} all exercise >1 chunk
    mol = gto.M(
        atom='''Na 0.0 0.0 0.0
                Na 2.1 0.0 0.0
                Na 0.0 2.3 0.0
                Na 1.0 1.0 2.0''',
        basis={'Na': basis_na}, ecp={'Na': ecp_na},
        cart=True, output='/dev/null')


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues(unittest.TestCase):
    def test_ip_batching_invariant(self):
        ref = cp.asnumpy(ecp.get_ecp_ip(mol, batch_size=1000))
        for bs in (1, 2, 3):
            got = cp.asnumpy(ecp.get_ecp_ip(mol, batch_size=bs))
            self.assertEqual(got.shape, ref.shape)
            self.assertAlmostEqual(abs(ref - got).max(), 0, 12, msg=f'bs={bs}')

    def test_ipip_batching_invariant(self):
        for ip_type in ('ipipv', 'ipvip'):
            ref = cp.asnumpy(ecp.get_ecp_ipip(mol, ip_type, batch_size=1000))
            for bs in (1, 2):
                got = cp.asnumpy(ecp.get_ecp_ipip(mol, ip_type, batch_size=bs))
                self.assertAlmostEqual(abs(ref - got).max(), 0, 12,
                                       msg=f'{ip_type} bs={bs}')

    def test_sum_helpers(self):
        ref_ip = ecp.get_ecp_ip(mol).sum(axis=0)
        got_ip = ecp.get_ecp_ip_sum(mol)
        self.assertEqual(got_ip.shape, ref_ip.shape)
        self.assertAlmostEqual(float(cp.abs(ref_ip - got_ip).max()), 0, 12)

        for ip_type in ('ipipv', 'ipvip'):
            ref = ecp.get_ecp_ipip(mol, ip_type).sum(axis=0)
            got = ecp.get_ecp_ipip_sum(mol, ip_type)
            self.assertAlmostEqual(float(cp.abs(ref - got).max()), 0, 12,
                                   msg=ip_type)

    def test_atom_subset_matches_full_rows(self):
        full = cp.asnumpy(ecp.get_ecp_ip(mol))          # rows == sorted ecp atoms
        all_atoms = sorted(set(int(a) for a in mol._ecpbas[:, gto.ATOM_OF]))
        subset = all_atoms[::2]                          # e.g. atoms 0, 2
        sub = cp.asnumpy(ecp.get_ecp_ip(mol, ecp_atoms=subset))
        self.assertEqual(sub.shape[0], len(subset))
        for r, atm in enumerate(subset):
            pos = all_atoms.index(atm)
            self.assertAlmostEqual(abs(full[pos] - sub[r]).max(), 0, 12,
                                   msg=f'atom {atm}')

    def test_loop_atom_ids_align(self):
        all_atoms = sorted(set(int(a) for a in mol._ecpbas[:, gto.ATOM_OF]))
        seen = []
        rows = []
        for batch, mat in ecp.loop_ecp_ip(mol, batch_size=2):
            self.assertEqual(mat.shape[0], len(batch))
            seen.extend(batch)
            rows.append(cp.asnumpy(mat))
        self.assertEqual(seen, all_atoms)
        stacked = np.concatenate(rows, axis=0)
        full = cp.asnumpy(ecp.get_ecp_ip(mol))
        self.assertAlmostEqual(abs(stacked - full).max(), 0, 12)

    def test_ipip_default_atom_order_is_sorted(self):
        # get_ecp_ipip used to default to set() iteration order; hess consumers
        # index its rows by enumerate(sorted(...)).  Guard the sorted default.
        all_atoms = sorted(set(int(a) for a in mol._ecpbas[:, gto.ATOM_OF]))
        by_default = cp.asnumpy(ecp.get_ecp_ipip(mol, 'ipipv'))
        explicit = cp.asnumpy(ecp.get_ecp_ipip(mol, 'ipipv', ecp_atoms=all_atoms))
        self.assertAlmostEqual(abs(by_default - explicit).max(), 0, 12)


if __name__ == '__main__':
    print('Tests for ECP-atom slicing / batching')
    unittest.main()
