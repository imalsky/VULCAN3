// CUDA twin of block_thomas_cpu.cc (VULCAN3_SOLVER=ffi on a GPU): the same
// block-Thomas factor and solve on [..., nz, ni, ni] blocks, with the same perm
// convention (perm[i] = source row of permuted row i). One thread block per
// batch element, the whole nz sweep inside the kernel. A zero pivot gives
// inf/nan like lax.linalg.lu, not an error. Build: python -m vulcan3.solver_fast --cuda.
// It takes sm_80+ and ni <= 96; solver_fast refuses a wider network before any
// launch (VULCAN3_SOLVER=fast takes any ni), and both handlers refuse it too.
//
// The factor, bt_gj_factor_kernel, holds each block in registers (ni <=
// kGjMaxNi = 96) and runs one exchange-form Gauss-Jordan sweep per layer: on
// the rows not yet pivoted its operations are the partial-pivot LU's, so it
// gives the CPU kernel's lu (partial-pivot row order) and perm, and the same
// sweep leaves inv(A'_j) for the next layer's Schur update, where the CPU
// kernel substitutes on the LU; beyond layer 0 the two differ by roundoff. The
// pivot step is on the kernel's critical path, so it reduces the warp's
// candidates with three REDUX operations (bt_warp_argmax) and takes the
// multipliers v / piv as products with the IEEE 1 / piv and two FMA
// corrections (bt_div_by_rcp), dividing only outside that sequence's range:
// the same winner and the same IEEE quotients as shuffle rounds and divisions.
//
// The solve launches 2 ni threads rounded up to a warp. It has two variants
// with bitwise equal results: two ni x ni double blocks of shared memory plus
// O(ni) (about 125 KB at ni = 89), or one (about 63 KB), which fits more blocks
// on an SM. A call takes two buffers when they fit the device and need no more
// waves than one (bt_choose); VULCAN3_BT_BUFFERS=1 or 2 forces a variant.
// Shared memory is opted in to the device maximum. The solve keeps each lane's
// substitution rows in registers and stages each layer's LU in shared memory
// with cp.async (bt_load_block), every thread's copies in flight at once.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "block_thomas_cuda.cu needs sm_80 or newer (REDUX, cp.async); set NVCC_ARCH"
#endif
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>

#include "block_thomas_common.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

using namespace vulcan_bt;

constexpr int kWarpSize = 32;
constexpr int kLaneMask = kWarpSize - 1;
constexpr unsigned kFullMask = 0xffffffffu;                         // every lane of a warp
constexpr int64_t kMaxGridX = std::numeric_limits<int32_t>::max();  // CUDA gridDim.x limit
// Rows per lane the solve's substitution holds in registers: ni <= kMaxNb
// kWarpSize = 96, the factor's kGjMaxNi. The solve kernel is compiled for
// every NB up to kMaxNb and launched with NB = ceil(ni / kWarpSize).
constexpr int kMaxNb = 3;
// Minimum resident blocks per SM in the NB solve kernel's __launch_bounds__
// (its maximum is 2 NB warps, the most threads bt_threads gives that NB). They
// hold NB 1-3 at 68-80 registers on sm_89 and sm_90 without spills, so the NB 3
// class (ni 65-96) keeps the 3 blocks per SM shared memory allows at ni 89 on
// sm_90.
constexpr int kSolveMinBlocks[kMaxNb + 1] = {0, 12, 6, 4};

// The Gauss-Jordan factor holds a kGjMaxNi x kGjMaxNi tile (the block, zero
// padded) in registers: thread (warp w, lane l) owns rows l + kWarpSize a
// (a < kGjRowSlots) and columns w + kGjWarps c (c < kGjColSlots). A column
// belongs to one warp, so the pivot search is shuffles only, and every warp
// holds every row, so each reads the pivot row from its own lane.
constexpr int kGjWarps = 8;
constexpr int kGjThreads = kGjWarps * kWarpSize;
constexpr int kGjRowSlots = 3;
constexpr int kGjColSlots = 12;
constexpr int kGjMaxNi = kWarpSize * kGjRowSlots;  // 96
static_assert(kGjWarps * kGjColSlots == kGjMaxNi, "the warps' columns must cover the tile");
static_assert(kMaxNb * kWarpSize == kGjMaxNi, "the solve takes the blocks the factor takes");
static_assert((kGjWarps & (kGjWarps - 1)) == 0, "column ownership is a power-of-two modulus");
static_assert(kGjColSlots % 2 == 0, "the pivot-row copy is read as double2");
// __launch_bounds__ minimum blocks per SM: two 256-thread blocks cap the
// kernel at 128 registers, which hold the 36-double tile without spills on
// sm_90 (and two blocks' shared memory fit at ni 96).
constexpr int kGjMinBlocks = 2;
// Pivot data of steps k - 1 (the deferred update), k and k + 1 (the
// lookahead) is live at once, so it rotates through three buffers.
constexpr int kGjPivotBufs = 3;
// A pivot candidate's key is (position << kGjRowBits) | row, so the smallest
// position wins a tie.
constexpr int kGjRowBits = 8;
static_assert(kGjMaxNi <= (1 << kGjRowBits), "rows must fit the pivot key");
constexpr int kGjNoKey = std::numeric_limits<int>::max();  // loses every tie
// Doubles of the next layer's diagonal block one thread prefetches to L2 per
// stride: 128 B, one L2 line.
constexpr int kGjPrefetchDoubles = 16;
// Shared memory of the Gauss-Jordan factor in doubles from the base: the
// pivot data at fixed offsets, then S (ni^2), which collects the LU in
// original-row order during the sweep and then holds inv(A'_j). The ints are
// the pivot row per buffer (padded to an even count), sperm and spos.
constexpr int kGjOffRowbuf = kGjPivotBufs * kGjMaxNi;              // after f
constexpr int kGjOffInfo = kGjOffRowbuf + kGjWarps * kGjColSlots;  // after rowbuf
constexpr int kGjOffInt = kGjOffInfo + 2 * kGjPivotBufs;           // after pinfo
constexpr int kGjProwInts = kGjPivotBufs + 1;
constexpr int kGjIntDoubles = (kGjProwInts + 2 * kGjMaxNi + 1) / 2;
constexpr int kGjOffS = (kGjOffInt + kGjIntDoubles + 1) / 2 * 2;  // 16-byte aligned
static_assert(kGjOffRowbuf % 2 == 0, "rowbuf is read as double2");
// Range of bt_div_by_rcp: |piv| within [2^-1000, 2^1000] (a normal reciprocal),
// |a| at least and |a y| within 2^+-960 (normal products, exact residuals).
// The solve (bt_lu_solve_warp0) multiplies by 1 / piv only in the piv range.
constexpr double kGjRcpPivMin = 0x1p-1000, kGjRcpPivMax = 0x1p1000;
constexpr double kGjRcpQuotMin = 0x1p-960, kGjRcpQuotMax = 0x1p960;

// x <- A^{-1} b from one block's LU, run by all lanes of warp 0 and no other
// warp. On entry the shared vector tmp holds b[perm], visible to warp 0; tmp is
// not written, and x (shared or global) must not overlap it. Lane l owns the
// rows r with r % kWarpSize == l and keeps them in registers, t[q] for row
// l + q kWarpSize (NB >= ceil(ni / kWarpSize) slots, indexed only by unrolled
// constants so they stay in registers), with the reciprocals of their U
// diagonal, taken up front and off the back substitution's serial chain; the
// only cross-lane traffic is the __shfl_sync broadcast of each finished x_i
// (uniform outer loops, no early return). No block barrier: the other warps
// prefetch meanwhile. The forward substitution applies the CPU lu_solve's
// subtractions in its order, the back substitution in reverse order, and
// x_i = t_i * (1 / u_ii) rounds twice where the CPU's t_i / u_ii rounds once.
// When any lane's |u_ii| is outside [2^-1000, 2^1000] (kGjRcpPiv*: zero,
// infinite and NaN pivots included, and those whose reciprocal overflows or
// is subnormal) the warp takes x_i = t_i / u_ii for every row, as the CPU
// does; a zero pivot gives inf/nan.
template <int NB>
__device__ void bt_lu_solve_warp0(const double* lu, int ni, const double* tmp, double* x, int lane) {
  double t[NB], rd[NB];
  bool fast = true;
#pragma unroll
  for (int q = 0; q < NB; ++q) {
    const int r = lane + kWarpSize * q;
    const double u = (r < ni) ? lu[r * ni + r] : 1.0;
    t[q] = (r < ni) ? tmp[r] : 0.0;
    rd[q] = (r < ni) ? 1.0 / u : 0.0;
    fast = fast && fabs(u) >= kGjRcpPivMin && fabs(u) <= kGjRcpPivMax;  // false for 0, inf, NaN
  }
  fast = __all_sync(kFullMask, fast);
  // Row i sits in slot i / kWarpSize of lane i % kWarpSize. The loops over that
  // slot are unrolled, so every index into t is a constant; a row below i is
  // in a slot >= i's, a row above in a slot <= i's.
#pragma unroll
  for (int qi = 0; qi < NB; ++qi) {  // unit-lower L: row i is final
    for (int i = kWarpSize * qi; i < min(ni, kWarpSize * (qi + 1)); ++i) {
      const double xi = __shfl_sync(kFullMask, t[qi], i & kLaneMask);
#pragma unroll
      for (int q = qi; q < NB; ++q) {
        const int r = lane + kWarpSize * q;
        if (r > i && r < ni) t[q] -= lu[r * ni + i] * xi;
      }
    }
  }
#pragma unroll
  for (int qi = NB - 1; qi >= 0; --qi) {  // upper U
    for (int i = min(ni, kWarpSize * (qi + 1)) - 1; i >= kWarpSize * qi; --i) {
      double xi = 0.0;
      if (lane == (i & kLaneMask)) {
        xi = fast ? t[qi] * rd[qi] : t[qi] / lu[i * ni + i];
        x[i] = xi;
      }
      xi = __shfl_sync(kFullMask, xi, i & kLaneMask);
#pragma unroll
      for (int q = 0; q <= qi; ++q) {
        const int r = lane + kWarpSize * q;
        if (r < i) t[q] -= lu[r * ni + i] * xi;
      }
    }
  }
}

// The Gauss-Jordan factor's shared memory (layout at kGjOff*).
struct BtGjSmem {
  double* f;       // kGjPivotBufs x kGjMaxNi multipliers: f[buf * kGjMaxNi + row]
  double* rowbuf;  // kGjWarps x kGjColSlots: each warp's copy of the pivot row
  double* pinfo;   // (piv, 1 / piv) per buffer
  int* prow;       // pivot row per buffer
  int* sperm;      // position -> original row (the partial-pivot perm)
  int* spos;       // original row -> position
  double* S;       // ni^2
};

__device__ __forceinline__ BtGjSmem bt_gj_carve(double* base) {
  BtGjSmem m;
  m.f = base;
  m.rowbuf = base + kGjOffRowbuf;
  m.pinfo = base + kGjOffInfo;
  m.prow = reinterpret_cast<int*>(base + kGjOffInt);
  m.sperm = m.prow + kGjProwInts;
  m.spos = m.sperm + kGjMaxNi;
  m.S = base + kGjOffS;
  return m;
}

__device__ __forceinline__ void bt_prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

// v[i] for a run-time i, by selects, so v stays in registers.
template <int N>
__device__ __forceinline__ double bt_pick(const double (&v)[N], int i) {
  double x = v[0];
#pragma unroll
  for (int q = 1; q < N; ++q)
    if (q == i) x = v[q];
  return x;
}

// The warp's best pivot candidate (largest best_abs, then smallest best_key),
// left in every lane. best_abs is -1 (no candidate, or a NaN) or |a| >= 0, so
// its bits plus one order the candidates as the doubles do: three REDUX
// operations take the largest image and the smallest key among its holders,
// the winner of five xor-shuffle rounds in about 150 cycles instead of about
// 390 on a GH200, on the pivot chain.
__device__ __forceinline__ void bt_warp_argmax(double& best_abs, double& best_val, int& best_key) {
  const unsigned long long img =
      (best_abs < 0.0) ? 0ull : static_cast<unsigned long long>(__double_as_longlong(best_abs)) + 1ull;
  const unsigned hi = static_cast<unsigned>(img >> 32), lo = static_cast<unsigned>(img);
  const unsigned max_hi = __reduce_max_sync(kFullMask, hi);
  const unsigned max_lo = __reduce_max_sync(kFullMask, hi == max_hi ? lo : 0u);
  const int key = __reduce_min_sync(kFullMask, (hi == max_hi && lo == max_lo) ? best_key : kGjNoKey);
  const int src = key & kLaneMask;  // a key's row is lane + kWarpSize r: its lane holds it as its best
  best_abs = __shfl_sync(kFullMask, best_abs, src);
  best_val = __shfl_sync(kFullMask, best_val, src);
  best_key = key;
}

// a / b from y = 1 / b (IEEE): q0 = a y and two FMA corrections with the
// exact residual a - b q. y correctly rounded leaves q1 within an ulp of a / b
// and the second correction rounds it correctly (Markstein), so the result is
// the IEEE quotient as long as nothing over- or underflows. `ok` says so: a = 0
// (q0 is then the signed zero a / b gives), or |a| and |q0| within the
// kGjRcpQuot range; the caller checks |b|. Five dependent FP64 operations,
// where a division is a longer sequence with a slow-path branch.
__device__ __forceinline__ double bt_div_by_rcp(double a, double b, double y, bool& ok) {
  const double q0 = a * y;
  const double r0 = fma(-q0, b, a);
  const double q1 = fma(r0, y, q0);
  const double r1 = fma(-q1, b, a);
  const double q2 = fma(r1, y, q1);
  const double aa = fabs(a), aq = fabs(q0);
  ok = (a == 0.0) || (aa >= kGjRcpQuotMin && aq >= kGjRcpQuotMin && aq <= kGjRcpQuotMax);
  return (a == 0.0) ? q0 : q2;
}

// Run by the whole warp that owns column kk; v holds column kk of the lane's
// rows, updated through step kk - 1. Chooses pivot kk with the CPU lu_inplace's
// rule (a row at a later position wins only on a strictly larger |a| than the
// row at position kk, the earliest position wins a tie, a NaN never wins), then
// publishes it in buffer nb: f (v / piv, the LU's multiplier; 0 for the pivot
// row and the padding), the pivot row and (piv, 1 / piv). The multipliers are
// bt_div_by_rcp's corrected products with the IEEE 1 / piv, bitwise v / piv;
// when any lane's operands leave its range (|piv| beyond about 2^+-1000, a
// zero, infinite or NaN pivot, |v| or |v / piv| beyond about 2^+-960) the warp
// divides instead. The L entries of
// column kk and the U diagonal go to S in original-row order, the position
// swap to sperm and spos.
__device__ __forceinline__ void bt_gj_pivot(const double (&v)[kGjRowSlots], int kk, int nb, int ni, int lane,
                                            const BtGjSmem& m) {
  int pos[kGjRowSlots];
  double best_abs = -1.0, best_val = 0.0;
  int best_key = kGjNoKey;
#pragma unroll
  for (int r = 0; r < kGjRowSlots; ++r) {
    const int row = lane + kWarpSize * r;
    pos[r] = (row < ni) ? m.spos[row] : -1;
    if (pos[r] > kk) {
      const double av = fabs(v[r]);
      const double cand = (av == av) ? av : -1.0;  // NaN never wins
      const int key = (pos[r] << kGjRowBits) | row;
      if (cand > best_abs || (cand == best_abs && key < best_key)) {
        best_abs = cand;
        best_val = v[r];
        best_key = key;
      }
    }
  }
  bt_warp_argmax(best_abs, best_val, best_key);
  const int rowk = m.sperm[kk];  // the row at position kk
  const double akk = __shfl_sync(kFullMask, bt_pick(v, rowk / kWarpSize), rowk & kLaneMask);
  const bool swap = best_abs > fabs(akk);  // -1 and NaN never beat |akk|
  const int ppos = swap ? (best_key >> kGjRowBits) : kk;
  const int prow = swap ? (best_key & ((1 << kGjRowBits) - 1)) : rowk;
  const double piv = swap ? best_val : akk;
  const double rpiv = 1.0 / piv;
  bool fast = fabs(piv) >= kGjRcpPivMin && fabs(piv) <= kGjRcpPivMax;  // false for 0, inf, NaN
  double f[kGjRowSlots];
#pragma unroll
  for (int r = 0; r < kGjRowSlots; ++r) {
    const int row = lane + kWarpSize * r;
    bool ok = true;
    f[r] = (row < ni && row != prow) ? bt_div_by_rcp(v[r], piv, rpiv, ok) : 0.0;
    fast = fast && ok;
  }
  if (!__all_sync(kFullMask, fast)) {  // outside the corrected range: IEEE divisions
#pragma unroll
    for (int r = 0; r < kGjRowSlots; ++r) {
      const int row = lane + kWarpSize * r;
      f[r] = (row < ni && row != prow) ? v[r] / piv : 0.0;
    }
  }
#pragma unroll
  for (int r = 0; r < kGjRowSlots; ++r) {
    const int row = lane + kWarpSize * r;
    m.f[nb * kGjMaxNi + row] = f[r];
    if (pos[r] >= kk && row != prow) m.S[row * ni + kk] = f[r];  // L, original row order
  }
  __syncwarp();  // every lane's sperm and spos reads before lane 0's swap
  if (lane == 0) {
    m.S[prow * ni + kk] = piv;  // U diagonal
    m.prow[nb] = prow;
    m.pinfo[2 * nb] = piv;
    m.pinfo[2 * nb + 1] = rpiv;
    if (swap) {
      m.sperm[kk] = prow;
      m.sperm[ppos] = rowk;
      m.spos[prow] = kk;
      m.spos[rowk] = ppos;
    }
  }
}

// Exchange update of step `step` (pivot data in buffer nb) on the warp's tile:
// the pivot row's lane publishes the row's values through step - 1 to rowbuf
// and its U entries (columns > step) to S; then a_ij -= f_i a_pj, the pivot
// row is scaled by -1 / piv, and the owner of column `step` sets that column
// to f, with 1 / piv in the pivot row. f is 0 in the pivot row, so the update
// leaves it for the scale.
__device__ __forceinline__ void bt_gj_update(double (&a)[kGjRowSlots][kGjColSlots], int step, int nb, int ni, int lane,
                                             int warp, const BtGjSmem& m) {
  const int prow = m.prow[nb];
  const double rcp = m.pinfo[2 * nb + 1];
  const int pl = prow & kLaneMask, pr = prow / kWarpSize;
  double f[kGjRowSlots];
#pragma unroll
  for (int r = 0; r < kGjRowSlots; ++r) f[r] = m.f[nb * kGjMaxNi + lane + kWarpSize * r];
  double* rb = m.rowbuf + warp * kGjColSlots;
  if (lane == pl) {
#pragma unroll
    for (int c = 0; c < kGjColSlots; ++c) {
      double x = a[0][c];
#pragma unroll
      for (int q = 1; q < kGjRowSlots; ++q)
        if (q == pr) x = a[q][c];
      rb[c] = x;
      const int col = warp + kGjWarps * c;
      if (col > step && col < ni) m.S[prow * ni + col] = x;  // U row of `step`
    }
  }
  __syncwarp();
  const double2* rb2 = reinterpret_cast<const double2*>(rb);
#pragma unroll
  for (int c2 = 0; c2 < kGjColSlots / 2; ++c2) {
    const double2 rv = rb2[c2];
#pragma unroll
    for (int r = 0; r < kGjRowSlots; ++r) {
      a[r][2 * c2] = fma(-f[r], rv.x, a[r][2 * c2]);
      a[r][2 * c2 + 1] = fma(-f[r], rv.y, a[r][2 * c2 + 1]);
    }
  }
  if (lane == pl) {
#pragma unroll
    for (int r = 0; r < kGjRowSlots; ++r)
      if (r == pr)
#pragma unroll
        for (int c = 0; c < kGjColSlots; ++c) a[r][c] = -(a[r][c] * rcp);
  }
  if (warp == step % kGjWarps) {
    const int ck = step / kGjWarps;
#pragma unroll
    for (int c = 0; c < kGjColSlots; ++c)
      if (c == ck)
#pragma unroll
        for (int r = 0; r < kGjRowSlots; ++r) a[r][c] = (lane + kWarpSize * r == prow) ? rcp : f[r];
  }
  __syncwarp();  // rowbuf is reused by the next update of this phase
}

// Factors one batch element: A'_0 = D_0, A'_j = D_j - (c_j b_{j-1}^T) .*
// inv(A'_{j-1}) with b = sup, c = sub, the LU of each A'_j and its perm
// written to lu and perm as the CPU kernel's. Per layer, pivot k of the
// Gauss-Jordan sweep with pivot (p, k): f_i = a_ik / a_pk, a_ij -= f_i a_pj
// (i != p, j != k), a_ik = f_i, a_pj = -a_pj (1 / a_pk), a_pk = 1 / a_pk. On
// the rows not yet pivoted that is the partial-pivot LU's trailing update (the
// CPU kernel's pivots and tie rule, f by IEEE division, the update an fma), so
// L and U are read off the published pivot rows and multipliers; after all ni
// pivots the tile holds inv(A'_j) with inv[pos(r)][sperm(c)] = a_rc.
// Lookahead: the owner of column k + 1 updates that column first, picks pivot
// k + 1 and publishes it while the other warps run update k, and defers its
// own update k to the next phase, off the pivot chain, so each pivot costs one
// block barrier. A layer ends with the LU store, inv(A'_j) into S and the
// Schur update A'_{j+1} = D_{j+1} - (c b^T) .* inv(A'_j) into the registers,
// barriers between them; D_{j+1} is prefetched to L2 while the layer is swept.
__global__ void __launch_bounds__(kGjThreads, kGjMinBlocks)
    bt_gj_factor_kernel(const double* __restrict__ diag, const double* __restrict__ sup, const double* __restrict__ sub,
                        double* __restrict__ lu, int32_t* __restrict__ perm, int nz, int ni) {
  extern __shared__ double bt_smem[];
  const BtGjSmem m = bt_gj_carve(bt_smem);
  const int blk = ni * ni;
  const int tid = threadIdx.x, lane = tid & kLaneMask, warp = tid / kWarpSize;
  const int64_t b = blockIdx.x;
  const double* D = diag + b * nz * static_cast<int64_t>(blk);
  const double* SU = sup + b * static_cast<int64_t>(nz - 1) * ni;
  const double* SB = sub + b * static_cast<int64_t>(nz - 1) * ni;

  double a[kGjRowSlots][kGjColSlots];
#pragma unroll
  for (int r = 0; r < kGjRowSlots; ++r)
#pragma unroll
    for (int c = 0; c < kGjColSlots; ++c) {
      const int row = lane + kWarpSize * r, col = warp + kGjWarps * c;
      a[r][c] = (row < ni && col < ni) ? D[row * ni + col] : 0.0;
    }

  for (int j = 0; j < nz; ++j) {
    if (j + 1 < nz)
      for (int off = tid * kGjPrefetchDoubles; off < blk; off += kGjThreads * kGjPrefetchDoubles)
        bt_prefetch_l2(D + static_cast<int64_t>(j + 1) * blk + off);
    for (int i = tid; i < kGjMaxNi; i += kGjThreads) {
      m.sperm[i] = i;
      m.spos[i] = i;
    }
    __syncthreads();
    if (warp == 0) {
      double v[kGjRowSlots];
#pragma unroll
      for (int r = 0; r < kGjRowSlots; ++r) v[r] = a[r][0];
      bt_gj_pivot(v, 0, 0, ni, lane, m);
    }
    __syncthreads();

    for (int k = 0, buf = 0; k < ni; ++k, buf = (buf == kGjPivotBufs - 1) ? 0 : buf + 1) {
      const int kn = k + 1, nbuf = (buf == kGjPivotBufs - 1) ? 0 : buf + 1;
      const int pbuf = (buf == 0) ? kGjPivotBufs - 1 : buf - 1;
      // the owner of column k ran pivot k's lookahead last phase and deferred
      // its update k - 1 to now
      if (k > 0 && warp == k % kGjWarps) bt_gj_update(a, k - 1, pbuf, ni, lane, warp, m);
      if (kn < ni && warp == kn % kGjWarps) {  // lookahead: column kn after update k
        const int prow = m.prow[buf];
        const double rcp = m.pinfo[2 * buf + 1];
        const int pl = prow & kLaneMask, pr = prow / kWarpSize;
        const int cn = kn / kGjWarps;
        double x[kGjRowSlots];
#pragma unroll
        for (int r = 0; r < kGjRowSlots; ++r) {
          x[r] = a[r][0];
#pragma unroll
          for (int c = 1; c < kGjColSlots; ++c)
            if (c == cn) x[r] = a[r][c];
        }
        const double rv = __shfl_sync(kFullMask, bt_pick(x, pr), pl);  // pivot row, column kn
        double v[kGjRowSlots];
#pragma unroll
        for (int r = 0; r < kGjRowSlots; ++r) {
          const double f = m.f[buf * kGjMaxNi + lane + kWarpSize * r];
          // the arithmetic bt_gj_update applies to this element
          v[r] = (lane + kWarpSize * r == prow) ? -(fma(-f, rv, x[r]) * rcp) : fma(-f, rv, x[r]);
        }
        bt_gj_pivot(v, kn, nbuf, ni, lane, m);
      } else {
        bt_gj_update(a, k, buf, ni, lane, warp, m);
      }
      __syncthreads();
    }

    // LU in partial-pivot row order (row i of lu = original row sperm[i]) and perm
    double* Lj = lu + (b * nz + j) * static_cast<int64_t>(blk);
    for (int e = tid; e < blk; e += kGjThreads) {
      const int i = e / ni, c = e - i * ni;
      Lj[e] = m.S[m.sperm[i] * ni + c];
    }
    int32_t* Pj = perm + (b * nz + j) * static_cast<int64_t>(ni);
    for (int i = tid; i < ni; i += kGjThreads) Pj[i] = m.sperm[i];
    __syncthreads();
    if (j + 1 == nz) break;
    {  // inv(A'_j) into S, row-major: inv[pos(row)][sperm(col)] = a[row][col]
      int rpos[kGjRowSlots];
#pragma unroll
      for (int r = 0; r < kGjRowSlots; ++r) {
        const int row = lane + kWarpSize * r;
        rpos[r] = (row < ni) ? m.spos[row] : -1;
      }
#pragma unroll
      for (int c = 0; c < kGjColSlots; ++c) {
        const int col = warp + kGjWarps * c;
        if (col < ni) {
          const int cs = m.sperm[col];
#pragma unroll
          for (int r = 0; r < kGjRowSlots; ++r)
            if (rpos[r] >= 0) m.S[rpos[r] * ni + cs] = a[r][c];
        }
      }
    }
    __syncthreads();
    {  // A'_{j+1} = D_{j+1} - (c b^T) .* inv(A'_j), c = sub[j], b = sup[j]
      const double* Dn = D + static_cast<int64_t>(j + 1) * blk;
      const double* cc = SB + static_cast<int64_t>(j) * ni;
      const double* bb = SU + static_cast<int64_t>(j) * ni;
      double cr[kGjRowSlots];
#pragma unroll
      for (int r = 0; r < kGjRowSlots; ++r) {
        const int row = lane + kWarpSize * r;
        cr[r] = (row < ni) ? cc[row] : 0.0;
      }
#pragma unroll
      for (int c = 0; c < kGjColSlots; ++c) {
        const int col = warp + kGjWarps * c;
        const double bs = (col < ni) ? bb[col] : 0.0;
#pragma unroll
        for (int r = 0; r < kGjRowSlots; ++r) {
          const int row = lane + kWarpSize * r;
          a[r][c] = (row < ni && col < ni) ? Dn[row * ni + col] - (cr[r] * bs) * m.S[row * ni + col] : 0.0;
        }
      }
    }
    __syncthreads();
  }
}

// Copies one layer's LU block into shared memory with threads t0, t0 + n0, ...
// The caller barriers. While warp 0 substitutes, the other warps call this as
// the prefetch, so the global latency hides behind the solve. Each thread
// issues all its 8-byte copies with cp.async and waits once, so its whole
// share is in flight together; a load-then-store loop keeps one load per
// thread in flight, and on a GH200 that latency bound the solve (78% of its
// stall samples, at the shared-memory store).
__device__ inline void bt_load_block(double* A, const double* L, int blk, int t0, int n0) {
  for (int i = t0; i < blk; i += n0) {
    const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(A + i));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(dst), "l"(__cvta_generic_to_global(L + i))
                 : "memory");
  }
  asm volatile("cp.async.commit_group;\n\tcp.async.wait_group 0;" ::: "memory");
}

__device__ inline void bt_swap(double*& a, double*& b) {
  double* t = a;
  a = b;
  b = t;
}

// Solves one rhs batch element on the factors bt_index maps it to: the forward
// sweep r'_j = r_j - c_j .* (A'_{j-1}^{-1} r'_{j-1}), then k_last = A'^{-1}
// r'_last and k_j = A'_j^{-1} (r'_j - b_j .* k_{j+1}); x holds r' until k
// overwrites it. Warp 0 runs every substitution. With two buffers the other
// warps prefetch the next layer's LU into the second one meanwhile (a one-warp
// block does both in turn); with one (kOneBuf) all threads load it after the
// substitution. Shared: the current and the prefetched LU (ni^2 doubles each;
// one with kOneBuf), t and u (ni each). Per layer the barriers are: u filled
// and the spare buffer free (one buffer: the layer loaded); the substitution
// done and the prefetch staged; in the forward sweep also r'_j complete (and
// the next layer loaded) before its permuted read.
template <bool kOneBuf, int NB>
__global__ void __launch_bounds__(2 * kWarpSize * NB, kSolveMinBlocks[NB])
    bt_solve_kernel(const double* __restrict__ lu, const int32_t* __restrict__ perm, const double* __restrict__ sup,
                    const double* __restrict__ sub, const double* __restrict__ rhs, double* __restrict__ x, int nz,
                    int ni, BtBcast s) {
  extern __shared__ double bt_smem[];
  const int blk = ni * ni;               // fits an int: the block lives in shared memory
  double* Ac = bt_smem;                  // the layer being solved
  double* An = kOneBuf ? Ac : Ac + blk;  // the layer being prefetched (two buffers)
  double* t = An + blk;                  // one layer's solution, and
  double* u = t + ni;                    // the permuted rhs the substitution consumes

  const int tid = threadIdx.x, nt = blockDim.x;
  const int warp = tid / kWarpSize, lane = tid & kLaneMask;
  const int64_t b = blockIdx.x;
  const int64_t band = static_cast<int64_t>(nz - 1) * ni;
  const double* L = lu + bt_index(b, s, kBtLu) * nz * blk;
  const int32_t* P = perm + bt_index(b, s, kBtPerm) * nz * ni;
  const double* S = sup + bt_index(b, s, kBtSup) * band;
  const double* C = sub + bt_index(b, s, kBtSub) * band;
  const double* R = rhs + b * nz * ni;
  double* X = x + b * nz * ni;
  const bool prefetches = (nt == kWarpSize) || (warp != 0);
  const int prefetch_tid = (nt > kWarpSize) ? tid - kWarpSize : tid;
  const int prefetch_nt = (nt > kWarpSize) ? nt - kWarpSize : nt;

  for (int i = tid; i < ni; i += nt) X[i] = R[i];
  bt_load_block(Ac, L, blk, tid, nt);
  __syncthreads();
  for (int i = tid; i < ni; i += nt) u[i] = X[P[i]];  // r'_0, permuted
  for (int j = 1; j < nz; ++j) {
    __syncthreads();  // u filled, An free
    if (!kOneBuf && prefetches) bt_load_block(An, L + static_cast<int64_t>(j) * blk, blk, prefetch_tid, prefetch_nt);
    if (warp == 0) bt_lu_solve_warp0<NB>(Ac, ni, u, t, lane);
    __syncthreads();  // t = A'_{j-1}^{-1} r'_{j-1}, An staged
    if (kOneBuf) bt_load_block(Ac, L + static_cast<int64_t>(j) * blk, blk, tid, nt);
    for (int i = tid; i < ni; i += nt) {
      const int64_t o = static_cast<int64_t>(j) * ni + i;
      X[o] = R[o] - C[static_cast<int64_t>(j - 1) * ni + i] * t[i];
    }
    __syncthreads();  // r'_j complete
    const int32_t* Pj = P + static_cast<int64_t>(j) * ni;
    for (int i = tid; i < ni; i += nt) u[i] = X[static_cast<int64_t>(j) * ni + Pj[i]];
    if (!kOneBuf) bt_swap(Ac, An);
  }
  __syncthreads();  // u filled, An free
  if (!kOneBuf && prefetches && nz > 1)
    bt_load_block(An, L + static_cast<int64_t>(nz - 2) * blk, blk, prefetch_tid, prefetch_nt);
  if (warp == 0)  // u already holds r'_last, so x may overwrite it
    bt_lu_solve_warp0<NB>(Ac, ni, u, X + static_cast<int64_t>(nz - 1) * ni, lane);
  __syncthreads();
  if (!kOneBuf) bt_swap(Ac, An);
  for (int j = nz - 2; j >= 0; --j) {
    if (kOneBuf) bt_load_block(Ac, L + static_cast<int64_t>(j) * blk, blk, tid, nt);
    // r'_j - b_j .* k_{j+1}, gathered straight into permuted order
    const int32_t* Pj = P + static_cast<int64_t>(j) * ni;
    for (int i = tid; i < ni; i += nt) {
      const int64_t o = static_cast<int64_t>(j) * ni + Pj[i];
      u[i] = X[o] - S[o] * X[o + ni];
    }
    __syncthreads();  // u filled, An free
    if (!kOneBuf && prefetches && j > 0)
      bt_load_block(An, L + static_cast<int64_t>(j - 1) * blk, blk, prefetch_tid, prefetch_nt);
    if (warp == 0) bt_lu_solve_warp0<NB>(Ac, ni, u, X + static_cast<int64_t>(j) * ni, lane);
    __syncthreads();  // k_j written, An staged
    if (!kOneBuf) bt_swap(Ac, An);
  }
}

// Threads per block of the solve: 2 ni rounded up to a warp (192 at ni = 89),
// at most 2 NB warps; warp 0 substitutes while the others prefetch.
int bt_threads(int ni) { return (2 * ni + kWarpSize - 1) / kWarpSize * kWarpSize; }

// Dynamic shared memory of each kernel; the layouts are in the kernels.
size_t bt_gj_factor_shmem(int ni) { return (kGjOffS + static_cast<size_t>(ni) * ni) * sizeof(double); }

size_t bt_solve_shmem(int ni, bool one_buf) {
  return ((one_buf ? 1 : 2) * static_cast<size_t>(ni) * ni + 2 * static_cast<size_t>(ni)) * sizeof(double);
}

using BtSolveKernel = void (*)(const double*, const int32_t*, const double*, const double*, const double*, double*, int,
                               int, BtBcast);

// The solve kernel with one or two buffers and NB register rows per lane.
BtSolveKernel bt_solve_kernel_for(bool one_buf, int nb) {
  static const BtSolveKernel k[2][kMaxNb] = {
      {bt_solve_kernel<false, 1>, bt_solve_kernel<false, 2>, bt_solve_kernel<false, 3>},
      {bt_solve_kernel<true, 1>, bt_solve_kernel<true, 2>, bt_solve_kernel<true, 3>}};
  return k[one_buf][nb - 1];
}

// Per device: the most dynamic shared memory every kernel may use and the SM
// count, or the CUDA error that stopped the opt-in.
struct BtOptIn {
  cudaError_t status;
  int limit;
  int sms;
};

// Dynamic shared memory above the 48 KB default needs a per-kernel opt-in.
// The factor and every solve kernel are opted in once per device, to the device's maximum, so a
// call changes no attribute: two differently sized calls cannot race on one,
// and only the launch happens while XLA records a command buffer. XLA makes
// the device's context current before calling a handler, so
// cudaFuncSetAttribute acts on `device`.
BtOptIn bt_opt_in(int device) {
  static std::mutex mu;
  static std::unordered_map<int, BtOptIn> done;
  std::lock_guard<std::mutex> lock(mu);
  const auto it = done.find(device);
  if (it != done.end()) return it->second;
  int optin = 0, sms = 0;
  cudaError_t r = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (r == cudaSuccess) r = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  int limit = optin;
  const void* kernels[1 + 2 * kMaxNb] = {reinterpret_cast<const void*>(bt_gj_factor_kernel)};
  for (int b = 0; b < 2; ++b)
    for (int nb = 1; nb <= kMaxNb; ++nb)
      kernels[1 + b * kMaxNb + nb - 1] = reinterpret_cast<const void*>(bt_solve_kernel_for(b, nb));
  for (const void* k : kernels) {
    cudaFuncAttributes a = {};
    if (r == cudaSuccess) r = cudaFuncGetAttributes(&a, k);
    // the opt-in maximum bounds static plus dynamic shared memory
    const int dyn = optin - static_cast<int>(a.sharedSizeBytes);
    if (r == cudaSuccess && dyn < limit) limit = dyn;
    if (r == cudaSuccess) r = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
  }
  done[device] = BtOptIn{r, limit, sms};
  return done[device];
}

// The device's opt-in (bt_opt_in) in *opt, or the error that stopped it.
ffi::Error bt_opt_in_checked(int device, BtOptIn* opt) {
  *opt = bt_opt_in(device);
  if (opt->status != cudaSuccess) {
    return ffi::Error::Internal(std::string("block-Thomas shared-memory opt-in failed: ") +
                                cudaGetErrorString(opt->status));
  }
  return ffi::Error::Success();
}

// An error unless `bytes` of dynamic shared memory fit the opted-in limit.
ffi::Error bt_shared_fits(size_t bytes, const BtOptIn& opt) {
  if (bytes > static_cast<size_t>(opt.limit)) {
    return ffi::Error::InvalidArgument("block-Thomas needs " + std::to_string(bytes) +
                                       " B of shared memory per block; this device allows " +
                                       std::to_string(opt.limit));
  }
  return ffi::Error::Success();
}

// VULCAN3_BT_BUFFERS, read once per process: unset or empty picks the
// variant per call (bt_choose); "1" or "2" forces the one- or two-buffer
// kernels, for the A/B and the byte-identity test.
const char* bt_forced_buffers() {
  static const char* const v = [] {
    const char* e = std::getenv("VULCAN3_BT_BUFFERS");
    return (e == nullptr) ? "" : e;
  }();
  return v;
}

// Sets *one_buf to the variant a call of `batch` blocks launches. Two buffers
// when they fit the device and need no more waves than one, a wave being the
// SM count times the blocks of that variant resident on an SM (the occupancy
// calculator, so the device's shared memory per SM, registers and block limit
// all count): at equal waves the two-buffer kernel is as fast or faster (it
// prefetches). Returns the error that forbids the launch, if any.
ffi::Error bt_choose(int device, int64_t batch, const void* two, size_t shmem2, const void* one, size_t shmem1,
                     int threads, bool* one_buf) {
  BtOptIn opt;
  ffi::Error err = bt_opt_in_checked(device, &opt);
  if (err.failure()) return err;
  const char* forced = bt_forced_buffers();
  if (std::strcmp(forced, "1") == 0 || std::strcmp(forced, "2") == 0) {
    *one_buf = forced[0] == '1';
  } else if (forced[0] != '\0') {
    return ffi::Error::InvalidArgument(std::string("VULCAN3_BT_BUFFERS=") + forced + ": expected 1, 2 or unset");
  } else if (shmem2 > static_cast<size_t>(opt.limit)) {
    *one_buf = true;
  } else {
    int per_sm2 = 0, per_sm1 = 0;
    cudaError_t r = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm2, two, threads, shmem2);
    if (r == cudaSuccess) r = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm1, one, threads, shmem1);
    if (r != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(r));
    const auto waves = [&](int per_sm) {  // no resident block: never chosen
      const int64_t w = static_cast<int64_t>(per_sm) * opt.sms;
      return (w > 0) ? (batch + w - 1) / w : std::numeric_limits<int64_t>::max();
    };
    *one_buf = waves(per_sm1) < waves(per_sm2);
  }
  return bt_shared_fits(*one_buf ? shmem1 : shmem2, opt);
}

// The launch's own error, if any; execution errors surface on the stream.
ffi::Error bt_launch_status() {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
  return ffi::Error::Success();
}

ffi::Error FactorImplCuda(cudaStream_t stream, int32_t device, ffi::Buffer<ffi::F64> diag, ffi::Buffer<ffi::F64> sup,
                          ffi::Buffer<ffi::F64> sub, ffi::ResultBuffer<ffi::F64> lu, ffi::ResultBuffer<ffi::S32> perm) {
  BtShape shape;
  ffi::Error err = bt_check_factor(diag, sup, sub, *lu, *perm, &shape);
  if (err.failure()) return err;
  if (shape.batch == 0 || shape.ni == 0) return ffi::Error::Success();
  if (shape.batch > kMaxGridX) return ffi::Error::InvalidArgument("batch exceeds the CUDA grid limit");
  if (shape.ni > kGjMaxNi) return ffi::Error::InvalidArgument("the CUDA factor takes ni <= 96");
  BtOptIn opt;
  err = bt_opt_in_checked(device, &opt);
  if (err.failure()) return err;
  const size_t shmem = bt_gj_factor_shmem(shape.ni);
  err = bt_shared_fits(shmem, opt);
  if (err.failure()) return err;
  bt_gj_factor_kernel<<<static_cast<unsigned int>(shape.batch), kGjThreads, shmem, stream>>>(
      diag.typed_data(), sup.typed_data(), sub.typed_data(), lu->typed_data(), perm->typed_data(), shape.nz, shape.ni);
  return bt_launch_status();
}

ffi::Error SolveImplCuda(cudaStream_t stream, int32_t device, ffi::Buffer<ffi::F64> lu, ffi::Buffer<ffi::S32> perm,
                         ffi::Buffer<ffi::F64> sup, ffi::Buffer<ffi::F64> sub, ffi::Buffer<ffi::F64> rhs,
                         ffi::ResultBuffer<ffi::F64> x) {
  BtShape shape;
  BtBcast s;
  ffi::Error err = bt_check_solve(lu, perm, sup, sub, rhs, *x, &shape, &s);
  if (err.failure()) return err;
  if (shape.batch == 0 || shape.ni == 0) return ffi::Error::Success();
  if (shape.batch > kMaxGridX) return ffi::Error::InvalidArgument("batch exceeds the CUDA grid limit");
  const int nb = (shape.ni + kWarpSize - 1) / kWarpSize;
  if (nb > kMaxNb) {
    return ffi::Error::InvalidArgument("block-Thomas solve holds at most " + std::to_string(kMaxNb * kWarpSize) +
                                       " rows per block in registers; ni = " + std::to_string(shape.ni));
  }
  const int threads = bt_threads(shape.ni);
  bool one_buf = false;
  err = bt_choose(device, shape.batch, reinterpret_cast<const void*>(bt_solve_kernel_for(false, nb)),
                  bt_solve_shmem(shape.ni, false), reinterpret_cast<const void*>(bt_solve_kernel_for(true, nb)),
                  bt_solve_shmem(shape.ni, true), threads, &one_buf);
  if (err.failure()) return err;
  bt_solve_kernel_for(
      one_buf, nb)<<<static_cast<unsigned int>(shape.batch), threads, bt_solve_shmem(shape.ni, one_buf), stream>>>(
      lu.typed_data(), perm.typed_data(), sup.typed_data(), sub.typed_data(), rhs.typed_data(), x->typed_data(),
      shape.nz, shape.ni, s);
  return bt_launch_status();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(VulcanBtFactorCuda, FactorImplCuda,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::DeviceOrdinal>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::S32>>(),
                              {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(VulcanBtSolveCuda, SolveImplCuda,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::DeviceOrdinal>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>(),
                              {ffi::Traits::kCmdBufferCompatible});
