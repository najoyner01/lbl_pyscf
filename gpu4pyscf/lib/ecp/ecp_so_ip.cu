/*
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
 */

/*
 * Bra-derivative of the spin-orbit ECP integrals ("ip1"):
 *
 *     gctr[C, a, x, i, j] = < d/dr_x  i | l_a  dU^SO(r) | j >,
 *         a in {x,y,z} (angular-momentum component of l_a),
 *         x in {x,y,z} (Cartesian derivative of the bra AO),
 *         C            = SO-ECP centre (reindexed by ecpbas[ECP_ATOM_ID]).
 *
 * This is `so_cart` (ecp_so.cu) with the bra-derivative recursion of
 * `type2_cart_ip1` (ecp_type2_ip.cu) grafted on:
 *
 *   - the ket projector index still carries the real antisymmetric L^a
 *     transform (transform_omega_lop, so_ang_matrix.cu) -- unchanged;
 *   - the bra Gaussian is differentiated the same way the scalar ECP `ip`
 *     kernel does it: evaluate the type-2 block once with the bra angular
 *     momentum raised (LI+1, radial order 1) and once lowered (LI-1, radial
 *     order 0), then map back with _li_down / _li_up.
 *
 * Unlike `so_cart`, the (i,j) antisymmetry is broken by the bra derivative
 * (< d i | .. | j > != +/- < d j | .. | i >), so NO transpose block is written
 * here -- the Python driver feeds a full (non-triangular) task list, exactly
 * like get_ecp_ip.
 *
 * Radial prefactor is identical to the scalar type-2 kernel (no extra 1/2 --
 * see the ecp_so.cu header).  Output layout:
 *     [n_ecp_atm, 3(a), 3(x), nao, nao], row-major, bra index = row.
 *
 * The complex/Pauli (spinor) assembly and the ket-transpose / translational
 * -invariance bookkeeping for the ECP-centre atom are done in Python
 * (grad/ghf.py:_soc_hcore_grad), mirroring grad/rhf.py's scalar-ECP path.
 */

// L^a on the (2*LC+1) projector index of `omega` (row-major, one (2*LC+1)
// vector per (bra-cart triple, lambda) row).  out[r, m'] = sum_m L[m',m] in[r, m].
// Identical to transform_omega_lop in ecp_so.cu; redeclared here so this file
// is self-contained w.r.t. include order (ecp_so.cu is included first in
// nr_ecp_driver.cu, so the symbol already exists -- keep names distinct).
__device__
static void so_ip_transform_omega_lop(double *out, const double *in,
                                      const int nrow, const int LC, const int a)
{
    const int dlc = 2*LC + 1;
    const double *L = _l_op[LC] + a*dlc*dlc;
    for (int r = threadIdx.x; r < nrow; r += blockDim.x){
        const double *win = in + (size_t)r*dlc;
        double *wout = out + (size_t)r*dlc;
        for (int mp = 0; mp < dlc; mp++){
            double s = 0.0;
            for (int m = 0; m < dlc; m++){
                s += L[mp*dlc + m] * win[m];
            }
            wout[mp] = s;
        }
    }
}

/*
 * SO type-2 block for a fixed bra/ket angular momentum, all three L^a
 * components, written into buf3 = [3(a)][nfi*nfj]  (nfi = (LI+1)(LI+2)/2).
 * `orderi`/`orderj` select the radial-derivative order on bra/ket (as in
 * type2_cart_kernel).  Uses dynamic shared memory laid out exactly like
 * so_cart:  rad_all | omegai | omegaj | omegaj_a | angi | angj.
 */
template <int orderi, int orderj> __device__
void so_type2_cart_kernel(double *buf3,
                const int LI, const int LJ, const int LC,
                const int ish, const int jsh, const int ksh,
                const int *ecpbas, const int *ecploc,
                const int *atm, const int *bas, const double *env)
{
    extern __shared__ double smem[];

    const double *ri = env + atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
    const double *rj = env + atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
    const int atm_id = ecpbas[ATOM_OF+ecploc[ksh]*BAS_SLOTS];
    const double *rc = env + atm[PTR_COORD+atm_id*ATM_SLOTS];

    double rca[3], rcb[3];
    rca[0] = rc[0] - ri[0];
    rca[1] = rc[1] - ri[1];
    rca[2] = rc[2] - ri[2];
    rcb[0] = rc[0] - rj[0];
    rcb[1] = rc[1] - rj[1];
    rcb[2] = rc[2] - rj[2];

    const int LIC1 = LI + LC + 1;
    const int LJC1 = LJ + LC + 1;
    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;

    const int omegai_sz = (LI+LC+2)/2 * (LI+1)*(LI+2)*(LI+3)/6 * (2*LC+1);
    const int omegaj_rows = (LJ+LC+2)/2 * ((LJ+1)*(LJ+2)*(LJ+3)/6);
    const int omegaj_sz = omegaj_rows * (2*LC+1);

    double *rad_all  = smem;
    double *omegai   = rad_all + (LI+LJ+1) * LIC1 * LJC1;
    double *omegaj   = omegai + omegai_sz;
    double *omegaj_a = omegaj + omegaj_sz;
    double *angi     = omegaj_a + omegaj_sz;
    double *angj     = angi + (LI+1)*nfi*LIC1;

    type2_facs_omega(omegai, LI, LC, rca);
    type2_facs_omega(omegaj, LJ, LC, rcb);
    __syncthreads();

    const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
    const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
    const double *ai = env + bas[PTR_EXP+ish*BAS_SLOTS];
    const double *aj = env + bas[PTR_EXP+jsh*BAS_SLOTS];
    const double *ci = env + bas[PTR_COEFF+ish*BAS_SLOTS];
    const double *cj = env + bas[PTR_COEFF+jsh*BAS_SLOTS];

    double radi[AO_LMAX+ECP_LMAX+orderi+1];
    const double dca = norm3d(rca[0], rca[1], rca[2]);
    type2_facs_rad<orderi>(radi, LI+LC, npi, dca, ci, ai);

    double radj[AO_LMAX+ECP_LMAX+orderj+1];
    const double dcb = norm3d(rcb[0], rcb[1], rcb[2]);
    type2_facs_rad<orderj>(radj, LJ+LC, npj, dcb, cj, aj);

    double ur = 0.0;
    for (int kbas = ecploc[ksh]; kbas < ecploc[ksh+1]; kbas++){
        ur += rad_part(kbas, ecpbas, env);
    }

    set_shared_memory(rad_all, (LI+LJ+1)*LIC1*LJC1);
    double root = 0.0;
    if (threadIdx.x < NGAUSS){
        root = r128[threadIdx.x];
    }
    for (int p = 0; p <= LI+LJ; p++){
        double *prad = rad_all + p*LIC1*LJC1;
        for (int i = 0; i <= LI+LC; i++){
        for (int j = 0; j <= LJ+LC; j++){
            block_reduce(radi[i]*radj[j]*ur, prad+i*LJC1+j);
        }}
        ur *= root;
    }
    __syncthreads();

    // same prefactor as the scalar type-2 kernel (no extra 1/2 -- see ecp_so.cu)
    const double fac = 16.0 * M_PI * M_PI * _common_fac[LI] * _common_fac[LJ];

    for (int ij = threadIdx.x; ij < 3*nfi*nfj; ij += blockDim.x){
        buf3[ij] = 0.0;
    }
    __syncthreads();

    for (int a = 0; a < 3; a++){
        so_ip_transform_omega_lop(omegaj_a, omegaj, omegaj_rows, LC, a);
        __syncthreads();

        double *buf = buf3 + (size_t)a*nfi*nfj;

        // (k+l)pq, k i m p, l j m q -> i j     (m: projector spherical index)
        for (int m = 0; m < 2*LC+1; m++){
            type2_ang(angi, LI, LC, rca, omegai   + m);
            type2_ang(angj, LJ, LC, rcb, omegaj_a + m);
            __syncthreads();

            for (int ij = threadIdx.x; ij < nfi*nfj; ij += blockDim.x){
                const int i = ij % nfi;
                const int j = ij / nfi;
                double s = 0.0;
                for (int k = 0; k <= LI; k++){
                for (int l = 0; l <= LJ; l++){
                    double *pangi = angi + k*nfi*LIC1 + i*LIC1;
                    double *pangj = angj + l*nfj*LJC1 + j*LJC1;
                    double *prad  = rad_all + (k+l)*LIC1*LJC1;
                    for (int p = 0; p < LIC1; p++){
                    for (int q = 0; q < LJC1; q++){
                        s += prad[p*LJC1+q] * pangi[p] * pangj[q];
                    }}
                }}
                buf[ij] += fac * s;
            }
            __syncthreads();
        }
    }
}

// General (runtime li,lj,lc) bra-derivative SO-ECP kernel.
__global__
void so_cart_ip1_general(double *gctr,
                const int LI, const int LJ, const int LC,
                const int *ao_loc, const int nao,
                const int *tasks, const int ntasks,
                const int *ecpbas, const int *ecploc,
                const int *atm, const int *bas, const double *env)
{
    const int task_id = blockIdx.x;
    if (task_id >= ntasks){
        return;
    }

    const int ish = tasks[task_id];
    const int jsh = tasks[task_id + ntasks];
    const int ksh = tasks[task_id + 2*ntasks];
    const int ioff = ao_loc[ish];
    const int joff = ao_loc[jsh];
    const int ecp_id = ecpbas[ECP_ATOM_ID+ecploc[ksh]*BAS_SLOTS];
    // output layout: [n_ecp_atm, 3(a), 3(x), nao, nao], bra index = row.
    gctr += (size_t)9*ecp_id*nao*nao + (size_t)ioff*nao + joff;

    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;

    // per-(a) 3 Cartesian-derivative sub-blocks -> 9 * nfi * nfj
    __shared__ double gctr_smem[9*NF_MAX*NF_MAX];
    for (int ij = threadIdx.x; ij < 9*nfi*nfj; ij += blockDim.x){
        gctr_smem[ij] = 0.0;
    }
    __syncthreads();

    constexpr int NFI_MAX = (AO_LMAX+2)*(AO_LMAX+3)/2;   // bra raised to LI+1
    constexpr int NFJ_MAX = (AO_LMAX+1)*(AO_LMAX+2)/2;
    __shared__ double buf3[3*NFI_MAX*NFJ_MAX];

    // + branch: bra angular momentum raised, radial order 1
    const int nfi1 = (LI+2) * (LI+3) / 2;
    so_type2_cart_kernel<1,0>(
        buf3, LI+1, LJ, LC, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
    for (int a = 0; a < 3; a++){
        _li_down(gctr_smem + (size_t)a*3*nfi*nfj, buf3 + (size_t)a*nfi1*nfj, LI, LJ);
    }
    __syncthreads();

    // - branch: bra angular momentum lowered, radial order 0
    if (LI > 0){
        const int nfi0 = LI * (LI+1) / 2;
        so_type2_cart_kernel<0,0>(
            buf3, LI-1, LJ, LC, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
        for (int a = 0; a < 3; a++){
            _li_up(gctr_smem + (size_t)a*3*nfi*nfj, buf3 + (size_t)a*nfi0*nfj, LI, LJ);
        }
        __syncthreads();
    }

    for (int ij = threadIdx.x; ij < nfi*nfj; ij += blockDim.x){
        const int i = ij % nfi;
        const int j = ij / nfi;
        for (int a = 0; a < 3; a++){
        for (int x = 0; x < 3; x++){
            double *g = gctr + (size_t)(a*3+x)*nao*nao;
            atomicAdd(g + (size_t)i*nao + j,
                      gctr_smem[(size_t)a*3*nfi*nfj + (size_t)x*nfi*nfj + ij]);
        }}
    }
    return;
}
