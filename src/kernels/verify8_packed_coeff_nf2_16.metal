#include "common.h"

// Exactly the original single-group arithmetic at a paired slice's odd edge.
NI_INLINE void v8pc_boundary_nf2(const device uint* w, const device ushort* s, const device ushort* b,
                                const device half* table, const device float* sums, uint N, uint K, uint n0,
                                uint g, uint lane, uint fm, uint fn, thread float2* acc) {
  const uint groups = K / 64, words = K / 8, q = fn / 2;
  const device uint32_t* wp[2];
  const device ushort* sp[2];
  const device ushort* bp[2];
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < 2; ++f) {
    const uint row = min(n0 + f * 8 + fm, N - 1);
    wp[f] = w + row * words + q * 2;
    sp[f] = s + row * groups;
    bp[f] = b + row * groups;
  }
  V8Ops<4, 2> c;
  v8_load<4, 2>(c, wp, sp, bp, table, sums, g, lane, fn);
  v8_compute<4, 2, false, 0, 1>(c, acc);
}

[[kernel]] void qmm8pc16_b4_nf2(
    const device uint* W0 [[buffer(0)]], const device ushort* S0 [[buffer(1)]],
    const device ushort* B0 [[buffer(2)]], device float* Y0 [[buffer(3)]],
    const device uint* W1 [[buffer(4)]], const device ushort* S1 [[buffer(5)]],
    const device ushort* B1 [[buffer(6)]], device float* Y1 [[buffer(7)]],
    const device uint* W2 [[buffer(8)]], const device ushort* S2 [[buffer(9)]],
    const device ushort* B2 [[buffer(10)]], device float* Y2 [[buffer(11)]],
    const device uint* W3 [[buffer(12)]], const device ushort* S3 [[buffer(13)]],
    const device ushort* B3 [[buffer(14)]], device float* Y3 [[buffer(15)]],
    const device half* table [[buffer(16)]], const device float* sums [[buffer(17)]],
    constant Qmm8Params& p [[buffer(18)]],
    const device uint* C0 [[buffer(21)]], const device uint* C1 [[buffer(22)]],
    const device uint* C2 [[buffer(23)]], const device uint* C3 [[buffer(24)]],
    const device uint* U0 [[buffer(25)]], const device uint* U1 [[buffer(26)]],
    const device uint* U2 [[buffer(27)]], const device uint* U3 [[buffer(28)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float2 red[2 * 16 * 32];
  const device uint* Wv[4] = {W0,W1,W2,W3};
  const device uint* Cv[4] = {C0,C1,C2,C3};
  const device uint* Uv[4] = {U0,U1,U2,U3};
  const device ushort* Sv[4] = {S0,S1,S2,S3};
  const device ushort* Bv[4] = {B0,B1,B2,B3};
  device float* Yv[4] = {Y0,Y1,Y2,Y3};
  uint si = 0;
  while (si + 1 < p.nseg && tg >= p.tg_begin[si + 1]) ++si;
  const uint cs = nsg / p.ks, kslice = sg % p.ks;
  const uint n0 = ((tg - p.tg_begin[si]) * cs + sg / p.ks) * 16;
  const uint N = p.N[si], groups = p.K / 64;
  const uint2 rc = v8_lane(lane);
  const uint gper = (groups + p.ks - 1) / p.ks;
  uint g0 = min(kslice * gper, groups);
  const uint g1 = min(g0 + gper, groups);
  float2 acc[2] = {float2(0.0f), float2(0.0f)};
  const uint first_block = (min(n0,N-1) / 16) * ((groups + 1) / 2);
  if ((g0 & 1u) && g0 < g1) {
    v8pc_boundary_nf2(Uv[si], Sv[si], Bv[si], table, sums, N, p.K, n0, g0, lane, rc.x, rc.y, acc);
    ++g0;
  }
  const uint gp = g0 + ((g1 - g0) & ~1u);
  v8pc_run_nf2(Wv[si], Cv[si], first_block, table, sums, g0, gp, lane, rc.x, rc.y, acc);
  if (gp < g1)
    v8pc_boundary_nf2(Uv[si], Sv[si], Bv[si], table, sums, N, p.K, n0, gp, lane, rc.x, rc.y, acc);
  if (p.ks > 1) {
    v8_reduce<2>(acc, red, p.ks, sg, lane);
    if (kslice) return;
  }
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < 2; ++f) {
    const uint n = n0 + f * 8 + rc.x;
    if (n >= N) continue;
    _Pragma("clang loop unroll(full)")
    for (int e = 0; e < 2; ++e) {
      const uint t = rc.y + e;
      if (t < p.R) {
        device float* dst = Yv[si] + t * p.y_ld[si] + n;
        *dst = p.add[si] ? *dst + acc[f][e] : acc[f][e];
      }
    }
  }
}
