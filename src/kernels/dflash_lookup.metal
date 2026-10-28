// Optional prompt-lookup proposals for an 8-row DFlash tree. Only committed tokens through the anchor are read.
// The target still verifies every proposal and kf_accept keeps its own samples. No target arithmetic changes.
#include "common.h"

struct DflLookupP {
  uint P, min_match, scan, cap;
};

// One 128-thread group; 32 B static threadgroup memory. The last 32K history endpoints are searched for the
// longest suffix match (at most 32 tokens), ties going to the latest endpoint. All seven continuation tokens
// must already exist at or before P. live[1] describes this draft; live[2] belongs to the completed verify and
// must survive the next drafting half's writes while the host observes the previous cycle.
[[kernel]] void dfl_lookup_tree(device int* tokens [[buffer(0)]], device int* tree [[buffer(1)]],
                                device uint* live [[buffer(2)]], constant DflLookupP& p [[buffer(3)]],
                                const device uint* gst [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  if (tid == 0) live[1] = 0;
  const uint P = p.P == 0xffffffffu ? gst[0] : p.P;
  if (p.min_match < 4 || p.min_match > 32 || p.cap <= 8 || P >= p.cap - 8 ||
      P < p.min_match + 6 || live[0] != 8) return;
  const uint first = max(p.min_match - 1, P > p.scan ? P - p.scan : 0u), last = P - 7;
  uint best_len = 0, best_end = 0;
  for (uint end = first + tid; end <= last; end += 128) {
    uint len = 0;
    while (len < 32 && len <= end && tokens[P - len] == tokens[end - len]) ++len;
    if (len >= p.min_match && (len > best_len || (len == best_len && end > best_end))) {
      best_len = len;
      best_end = end;
    }
  }
  const uint len = simd_max(best_len), end = simd_max(best_len == len ? best_end : 0u);
  threadgroup uint lengths[4], endpoints[4];
  if (lane == 0) {
    lengths[sg] = len;
    endpoints[sg] = end;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid != 0) return;
  best_len = best_end = 0;
  for (uint i = 0; i < 4; ++i)
    if (lengths[i] > best_len || (lengths[i] == best_len && endpoints[i] > best_end)) {
      best_len = lengths[i];
      best_end = endpoints[i];
    }
  if (best_len < p.min_match) return;
  for (uint r = 0; r < 8; ++r) {
    if (r) tokens[P + r] = tokens[best_end + r];
    tree[kKfPar + r] = int(r) - 1;
    tree[kKfDep + r] = int(r);
    tree[kKfMask + r] = int((1u << (r + 1)) - 1);
    tree[kKfLq + r] = 0;
    tree[kKfScore + r] = 0;
  }
  // These proposals have no neural probabilities. The host skips this cycle's budget/calibration observation.
  for (uint n = 0; n <= kKfMaxRows; ++n) tree[kKfE + n] = as_type<int>(n <= 8 ? float(n) : 0.0f);
  tree[kKfN] = tree[kKfNodes] = tree[kKfCap] = 8;
  tree[kKfJump] = 0;
  live[1] = best_len;
}

// Ordered after kf_accept and before the next drafting half. The next lookup only changes live[1].
[[kernel]] void dfl_lookup_record(device uint* live [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  if (tid == 0) live[2] = live[1];
}
