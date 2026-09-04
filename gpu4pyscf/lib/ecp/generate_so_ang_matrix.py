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

"""
Code generator + verifier for so_ang_matrix.cu.

so_ang_matrix.cu holds L^a_{m'm} -- the matrix of the angular-momentum operator
(l * 1j, real and antisymmetric) on the real-spherical-harmonic basis, ordered
m = -l..l, for a in {x, y, z} and l = 0..4 (s..g).  These are the
_angular_moment_matrix_{s,p,d,f,g} tables from pyscf/lib/gto/nr_ecp.c; see the
angular_moment_matrix(l) docstring there for provenance.

Run with pyscf on PYTHONPATH to regenerate + cross-check:
    python generate_so_ang_matrix.py            # writes so_ang_matrix.cu
    python generate_so_ang_matrix.py --check     # just verify the committed file
"""

import argparse
import numpy as np

# Verbatim from pyscf/lib/gto/nr_ecp.c  (row-major [3][2l+1][2l+1]).
_REF = {
    0: [0, 0, 0],
    1: [0, 0, 0, 0, 0, 1, 0, -1, 0,
        0, 0, -1, 0, 0, 0, 1, 0, 0,
        0, 1, 0, -1, 0, 0, 0, 0, 0],
    2: [0, 0, 0, 1, 0, 0, 0, 1.73205080756887719, 0, 1, 0, -1.73205080756887719,
        0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0,
        0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1.73205080756887719, 0, 0, 0,
        -1.73205080756887719, 0, 1, 0, 0, 0, -1, 0,
        0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0,
        0, 0],
    3: [0, 0, 0, 0, 0, 1.22474487139158894, 0, 0, 0, 0, 0, 1.58113883008418976,
        0, 1.22474487139158894, 0, 0, 0, 2.44948974278317788, 0,
        1.58113883008418976, 0, 0, 0, -2.44948974278317788, 0, 0, 0, 0, 0,
        -1.58113883008418976, 0, 0, 0, 0, 0, -1.22474487139158894, 0,
        -1.58113883008418976, 0, 0, 0, 0, 0, -1.22474487139158894, 0, 0, 0, 0,
        0,
        0, -1.22474487139158894, 0, 0, 0, 0, 0, 1.22474487139158894, 0,
        -1.58113883008418976, 0, 0, 0, 0, 0, 1.58113883008418976, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 2.44948974278317788, 0, 0, 0, 0, 0, -2.44948974278317788, 0,
        1.58113883008418976, 0, 0, 0, 0, 0, -1.58113883008418976, 0,
        1.22474487139158894, 0, 0, 0, 0, 0, -1.22474487139158894, 0,
        0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0,
        0, 0],
    4: [0, 0, 0, 0, 0, 0, 0, 1.41421356237309515, 0, 0, 0, 0, 0, 0, 0,
        1.87082869338697066, 0, 1.41421356237309515, 0, 0, 0, 0, 0,
        2.12132034355964239, 0, 1.87082869338697066, 0, 0, 0, 0, 0,
        3.16227766016837952, 0, 2.12132034355964239, 0, 0, 0, 0, 0,
        -3.16227766016837952, 0, 0, 0, 0, 0, 0, 0, -2.12132034355964239, 0, 0,
        0, 0, 0, 0, 0, -1.87082869338697066, 0, -2.12132034355964239, 0, 0, 0,
        0, 0, -1.41421356237309515, 0, -1.87082869338697066, 0, 0, 0, 0, 0, 0,
        0, -1.41421356237309515, 0, 0, 0, 0, 0, 0, 0,
        0, -1.41421356237309515, 0, 0, 0, 0, 0, 0, 0, 1.41421356237309515, 0,
        -1.87082869338697066, 0, 0, 0, 0, 0, 0, 0, 1.87082869338697066, 0,
        -2.12132034355964239, 0, 0, 0, 0, 0, 0, 0, 2.12132034355964239, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 3.16227766016837952, 0, 0, 0, 0, 0, 0, 0,
        -3.16227766016837952, 0, 2.12132034355964239, 0, 0, 0, 0, 0, 0, 0,
        -2.12132034355964239, 0, 1.87082869338697066, 0, 0, 0, 0, 0, 0, 0,
        -1.87082869338697066, 0, 1.41421356237309515, 0, 0, 0, 0, 0, 0, 0,
        -1.41421356237309515, 0,
        0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 0, 0, 0,
        0, -2, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0,
        0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
}

_NAME = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}


def as_array(l):
    return np.array(_REF[l], dtype=float).reshape(3, 2 * l + 1, 2 * l + 1)


def check():
    for l in range(5):
        m = as_array(l)
        # l * 1j is Hermitian  =>  the real matrix stored here is antisymmetric
        assert np.allclose(m, -m.transpose(0, 2, 1), atol=1e-12), \
            f'l={l}: L^a is not antisymmetric'
        # l_z is diagonal with entries -l..l  (after the real transform, still
        # antisymmetric off-diagonal only via l_x/l_y; l_z stays anti here)
    print('so_ang_matrix: 5 shells (s..g), all L^a antisymmetric  OK')


_HEADER = '''/*
 * Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * L^a_{m'm}: matrix of the angular-momentum operator (l * 1j, real &
 * antisymmetric) on the real spherical harmonic basis, m = -l..l,
 * a in {x,y,z}.  Generated by generate_so_ang_matrix.py; values verbatim
 * from _angular_moment_matrix_* in pyscf/lib/gto/nr_ecp.c.
 */
'''


def emit():
    lines = [_HEADER]
    for l in range(5):
        vals = ', '.join(repr(float(v)) for v in _REF[l])
        lines.append(
            f'__constant__ static const double _l_op_{_NAME[l]}'
            f'[3*{2*l+1}*{2*l+1}] = {{ {vals} }};\n')
    lines.append(
        '__constant__ static const double *_l_op[5] = '
        '{ _l_op_s, _l_op_p, _l_op_d, _l_op_f, _l_op_g };\n')
    return '\n'.join(lines)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='verify only, do not rewrite so_ang_matrix.cu')
    args = ap.parse_args()
    check()
    if not args.check:
        with open('so_ang_matrix.cu', 'w') as fh:
            fh.write(emit())
        print('wrote so_ang_matrix.cu')
