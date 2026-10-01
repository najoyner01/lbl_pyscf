# RC1 — the exchange kernel runs an nprim⁴ serial loop per shell quartet

Expanded explanation of root cause 1 in `V2_REVISION_PLAN.md` §3, with the
chemistry spelled out. Companion to `V2_REVISION_PLAN.md` (which carries the
measurements and the P1 remediation plan) and `STRATEGY.md`.

Status date: 2026-10-01. No code has been changed; this is background for the
plan.

---

## 1. The chemistry: what a "contracted" basis function is

A Gaussian basis function is not one Gaussian. It is a fixed linear combination
of **primitive** Gaussians:

$$\chi_\mu(\mathbf r) = \sum_{p}^{n_{\text{prim}}} c_{p}\, Y_{lm}\, e^{-\alpha_p |\mathbf r - \mathbf A|^2}$$

The reason is that a single Gaussian is a poor atomic orbital — it has zero
slope at the nucleus where the true wavefunction has a cusp, and it decays as
$e^{-\alpha r^2}$ where the true one decays as $e^{-\zeta r}$. Both ends are
fixed by stacking primitives: very steep ones ($\alpha \sim 10^4$) to build the
cusp, shallow ones ($\alpha \sim 0.1$) to build the tail. The coefficients $c_p$
are then *frozen* at their atomic-SCF values, so the molecular calculation has
one variational parameter per contracted function instead of $n_{\text{prim}}$.

This is a CPU-era trade: integral cost proportional to the primitive count, in
exchange for a much smaller matrix dimension (smaller `eigh`, smaller DIIS,
smaller everything downstream).

**Where the deep contractions live.** The actual AgCl/def2-TZVP shell structure:

| | shell | nprim | exponent range |
|---|---|---|---|
| Cl | s | **7** | 6.95×10⁴ → 28.9 |
| Cl | s | 3 | 127 → 7.7 |
| Cl | p | **5** | 667 → 7.3 |
| Ag | p | 4 | 13.2 → 0.98 |
| Ag | d | 4 | 25.8 → 1.19 |

The 7-primitive Cl s shell is the **1s core orbital**. Chlorine in def2 is
all-electron — no ECP — so the basis must resolve a 1s orbital in a $Z=17$
nuclear potential. The cusp condition forces an exponent of ~7×10⁴ (a Gaussian
whose radial maximum sits at 0.003 bohr), and six more primitives spanning three
decades are needed to bridge out to the 3s valence region. The 5-primitive p
shell is 2p, same story.

Silver is the opposite case, and it is instructive: def2 puts a 28-electron ECP
on Ag, so the 1s–3d core is gone, replaced by a smooth pseudopotential. Ag's
deepest contraction is only 4 primitives. **The ECP that makes Ag affordable is
exactly what Cl does not have.** In an inorganic complex it is frequently the
light main-group ligand atoms — Cl, S, P — that carry the deep contractions, not
the metal. (4d/5d metals with small-core ECPs, and 3d metals with no ECP at all,
do contribute.)

## 2. Why contraction depth is a quartic cost in the ERI

A two-electron integral over contracted functions expands into a quadruple sum
over primitives:

$$(\mu\nu|\lambda\sigma) = \sum_{p}^{n_i}\sum_{q}^{n_j}\sum_{r}^{n_k}\sum_{s}^{n_l} c_p c_q c_r c_s \,(p q | r s)$$

Every term is a separate primitive integral with its own Gaussian product
centers and its own Rys roots. For the (Cl s | Cl s | Cl s | Cl s) quartet with
$n=7$ on all four indices, that is $7^4 = 2401$ primitive integrals accumulated
into **one** number.

This is not wasted work in an absolute sense — those 2401 primitive integrals
must be evaluated however the calculation is organized. The question is only
*where the loop lives*, and that is where CPU and GPU part ways.

On a CPU this organization is a clear win. The inner primitive loop reuses the
shell-pair geometry, the Rys root tables and the angular-momentum recursion
scaffolding; everything stays in L1; and it emits one contracted integral
instead of 2401, so digestion into the Fock matrix is 2401× cheaper. Pople,
Head-Gordon, Rys — the classical ERI literature is built around keeping that
loop tight and innermost.

## 3. Why that same loop is poison on a GPU

`scf/jk.py:421`:

```python
mol = self.sorted_mol = SortedGTO.from_mol(
    self.mol, decontract=True, diffuse_cutoff=0.3)
```

`decontract=True` sounds like it undoes contraction, but `diffuse_cutoff=0.3`
restricts it. In `gto/mole.py:1184-1205`, the `nctr == 1` branch splits a shell
into (one segment holding every primitive with $\alpha > 0.3$) + (one shell per
primitive with $\alpha \le 0.3$):

```python
compact_idx = np.where(exps > diffuse_cutoff)[0]
diffuse_idx = np.where(exps <= diffuse_cutoff)[0]
...
shells[:nsegment, NPRIM_OF] = nprim_compact   # all compact prims stay in ONE segment
shells[nsegment:, NPRIM_OF] = 1               # each diffuse prim becomes its own shell
```

Cl's 1s has *all seven* primitives above 0.3 (the smallest is 28.9), so nothing
is split; it stays a 7-primitive segment. Across the molecule the cutoff removes
only the handful of genuine valence-tail functions.

Those surviving segments feed straight into the Rys kernel's loop structure
(`lib/gvhf-rys/rys_contract_k.cu:195-221`):

```c
for (int klp = 0; klp < kprim*lprim; ++klp) {
    __syncthreads();
    ...
    for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
        __syncthreads();
```

The whole quadruple primitive sum is **serial inside a single thread block**,
with a `__syncthreads()` barrier on every iteration of both loops. The barrier
is structurally necessary as written: the $g_x, g_y, g_z$ Rys intermediates live
in shared memory and are rebuilt from scratch for each primitive quartet, while
the Cartesian output components are spread across threads by `gout_id`. No
thread may start overwriting the shared scratch until every thread has finished
reading the previous primitive's copy.

So the GPU extracts parallelism across *angular momentum components and shell
quartets*, and **none** across primitives.

The measured scaling — `(ss|ss)` 1×1 pairs, 0.021 ms at nprim⁴=1 vs 9.5 ms at
nprim⁴=2401 — works out to **~4 µs per primitive iteration**, for an (ss|ss)
primitive integral that is a single Boys function and a multiply. That is pure
barrier and shared-memory-traffic latency; the floating-point work is invisible.

The fix in P1 is to stop hiding the parallelism: `diffuse_cutoff=1e200` fully
decontracts, so every primitive becomes its own shell and the loop trip count
collapses to 1. The identical 2401 primitive integrals are still computed — but
as 2401 **independent** shell quartets the scheduler can spread over the whole
device, with the contraction coefficients reapplied in the
`coeff.T @ M @ coeff` back-transform the sorted-basis machinery already
performs. Same arithmetic, same answer (K agrees to ≤5×10⁻¹¹), different
schedule.

**The J engine already does this.** `j_engine.py:100` uses
`diffuse_cutoff=1e200`. The 5–20× J/K gap in the measured table is not an
algorithmic difference between Coulomb and exchange — it is the same engine
family with two different contraction policies.

## 4. Why exchange and not Coulomb, physically

This explains the hybrid-vs-GGA split in the ladder.

The Coulomb matrix is a **local classical functional** of the total density:

$$J_{\mu\nu} = \sum_{\lambda\sigma} (\mu\nu|\lambda\sigma) D_{\lambda\sigma}$$

The density indices $\lambda\sigma$ are confined to the ket pair. $D$ can be
contracted into the ket charge distribution *once*, as a set of Hermite
multipole coefficients, after which J is a classical electrostatic interaction
between $N^2$ pair distributions. That is the McMurchie–Davidson J-engine.
Primitives are cheap there because the contraction over them happens before the
expensive long-range part.

Exchange is **nonlocal**:

$$K_{\mu\nu} = \sum_{\lambda\sigma} (\mu\lambda|\sigma\nu) D_{\lambda\sigma}$$

The density indices are split — one in the bra, one in the ket. There is no pair
distribution to collapse $D$ into; the four indices stay genuinely coupled and
each shell quartet must be visited individually. That is the structural reason K
goes through the Rys quartet kernel at all, and therefore the reason it inherits
that kernel's primitive loop.

Which closes the loop on the ladder data: PBE needs only J, so it never touches
this path. B3LYP (20 % exact exchange) and PBE0 (25 %) need K every cycle. Hence
**AgCl/TZVP: PBE 1.5 s vs B3LYP 12.6 s on GPU, but 0.39 vs 0.39 s on CPU.** On
the CPU J and K cost the same, because the CPU is happy with the contracted
loop. The entire hybrid penalty is this one scheduling decision.

## 5. The second-order damage: group fragmentation

Keeping shells contracted has a side effect that is arguably as expensive as the
serial loop, and is not obvious at first glance.

The CUDA kernels are templated on angular momentum and parameterized by
primitive count, so shells are bucketed into **(l, nprim) groups**, and one
kernel is launched per *quartet of groups*. For AgCl/def2-TZVP:

| policy | groups | shells |
|---|---|---|
| `diffuse_cutoff=0.3` (shipped) | 11 — s:{1,2,3,7}, p:{1,4,5}, d:{1,2,4}, f:{1} | 27 |
| `diffuse_cutoff=1e200` | **4** — s:{1}, p:{1}, d:{1}, f:{1} | 48 |

The task loop at `jk.py:520-524` is

```python
tasks = ((i,j,k,l) for i in range(n_groups) for j in range(i+1)
                   for k in range(i+1) for l in range(k+1))
```

i.e. $\sum_{m=1}^{n} m^2(m+1)/2$ launches. For $n=11$ that is exactly **2431**;
for $n=4$, exactly **65** — matching the figures in the plan.

The chemistry drives this directly: contraction depth is a property of the
element and the shell, so a basis with varied contraction depths shatters into
many small groups, and the launch count grows as roughly the **fourth power** of
that variety. Decontracting makes every shell look identical except for $l$, so
the group count floors at $l_{\max}+1$.

Three consequences, all visible in the measured data:

- **Launch overhead.** 2431 launches × ~5–10 µs is 12–25 ms of pure dispatch per
  K build, before any integral is computed.
- **Occupancy collapse.** Those launches partition a fixed quartet count into
  ever-finer slices. A group quartet such as (Cl-s7, Ag-d4, Cl-p5, Ag-s2) may
  contain a handful of shell quartets — a few blocks on a 108-SM A100. Mean 5.6
  of 108 SMs active: the GPU is ~95 % idle, and what is running is
  barrier-bound.
- **Setup cost as $n_{\text{groups}}^2$.** `_cache_q_cond_and_non0pairs` builds
  screening tables per (i,j) group *pair*: $\binom{11}{2}+11 = 66$ pairs instead
  of 10, × 8 `cp.where` each = the 528 tiny kernels of RC2. Hence
  `_VHFOpt.build` dropping 98→20 ms and 223→30 ms in the decontracted rows.

## 6. Why the 0.3 cutoff exists, and when it is right

It is not arbitrary, and the fix should not simply delete it.

$\alpha = 0.3$ bohr⁻² is a valence/diffuse boundary: the radial maximum of such
a Gaussian sits at $1/\sqrt{2\alpha} \approx 1.3$ bohr ≈ 0.7 Å — about a bond
half-length.

The relevant physics is the **Gaussian product theorem**, visible as `Kcd` in
the kernel:

$$e^{-\alpha r_A^2}e^{-\beta r_B^2} = \exp\!\left(-\tfrac{\alpha\beta}{\alpha+\beta}R_{AB}^2\right)e^{-(\alpha+\beta)r_P^2}$$

The prefactor $K_{ab} = \exp[-\tfrac{\alpha\beta}{\alpha+\beta}R_{AB}^2]$ is what
makes direct SCF scale sub-quartically. If both primitives are **compact** (large
$\alpha,\beta$), the reduced exponent is large and $K_{ab}$ dies within a bond
length — the pair is negligible beyond nearest neighbours and Schwarz screening
kills it. If both are **diffuse** ($\alpha,\beta \lesssim 0.3$), the reduced
exponent is small, $K_{ab} \approx 1$ out to many Ångström, and the pair
survives screening against essentially every other pair in the molecule.

The two classes therefore behave oppositely under decontraction:

- Splitting a **compact** segment into $n$ primitives creates $n^2$ shell pairs,
  but nearly all of them are still short-ranged and still screened out. The pair
  count grows locally and boundedly.
- Splitting a **diffuse** function multiplies the number of long-range pairs,
  and those are exactly the pairs that survive screening and pair with
  everything. $n_{\text{pairs}}$ grows superlinearly in system size.

Hence the shipped policy: *decontract the diffuse tail (one shell per primitive,
where it buys parallelism cheaply), keep the compact core contracted (where
decontracting would inflate the pair list).* That is the right call for a
500-atom organic molecule with C/H/N/O in def2-SVP, the workload gpu4pyscf was
tuned on — there the compact segments are shallow (C 1s is 3–6 primitives) and
there are hundreds of atoms, so $n_{\text{prim}}^4$ stays small while
$n_{\text{pairs}}$ is the binding constraint.

It inverts on a 2-atom inorganic complex. AgCl/TZVP has 27 shells; full
decontraction takes it to 48 shells → 1176 shell pairs, which is nothing — the
GPU is starving either way. Meanwhile the compact segments are 7 and 5
primitives deep, so $n_{\text{prim}}^4$ is the binding constraint by orders of
magnitude. The policy is optimizing the wrong term.

This is why P1 is framed as a **molecule-dependent choice** rather than a
constant change: switch on `max(nprim)^4 × n_shell_pairs` (or just `nbas`),
decontract fully when the pair count can absorb it, and keep the 0.3 policy for
the large-organic regime it was built for. The plan's validation requirement —
"organic set must not regress" — is precisely the guard on that boundary.

## 7. The one-paragraph version

> Basis sets bundle many primitive Gaussians into each basis function to capture
> the nuclear cusp cheaply — a CPU-era optimization. Chlorine, having no ECP in
> def2, needs 7 primitives spanning three decades for its 1s shell. gpu4pyscf's
> exchange engine keeps those bundles intact (`diffuse_cutoff=0.3`), so a single
> Cl (ss|ss) quartet expands into a 7⁴ = 2401-iteration loop that runs
> **serially inside one GPU thread block with a barrier per iteration** — 4 µs
> each, 9.5 ms total, for work the GPU could do in microseconds if the
> iterations ran side by side. Exchange is affected and Coulomb is not, because
> K's nonlocal index structure forces it through the quartet kernel while the
> J-engine collapses the density first — and the J path already decontracts
> fully. The same choice also shatters the basis into 11 (l,nprim) groups
> instead of 4, which becomes 2431 kernel launches per K build instead of 65 and
> leaves 5.6 of 108 SMs busy. Fully decontracting computes the identical
> integrals (K agrees to 5×10⁻¹¹) with the primitive sum exposed as independent
> parallel work: AgCl/TZVP B3LYP goes 12.7 s → 1.75 s.

**Headline for a non-specialist audience:** *the GPU code inherited a CPU
optimization, and on a GPU that optimization is a serialization.*

## 8. A framing caveat

The ladder table makes this look like a transition-metal problem, and
`V2_REVISION_PLAN.md` §3 says "4d/5d metals and Cl". The basis data says the
metal is often the innocent party: Ag's deepest contraction is 4 primitives
precisely because its ECP removed the core that would have required deep ones.
The worst offender in AgCl is the chlorine.

The accurate framing is **"all-electron atoms with deep core contractions"**,
which in this benchmark set means the main-group ligands plus the lighter or
large-core-free metals. That is more defensible than "transition metals are
slow", and it also predicts where the fix will and will not help.
