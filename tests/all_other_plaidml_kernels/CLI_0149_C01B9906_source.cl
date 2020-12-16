#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 19712 1 1
// lid: 256 1 1
// Original:
// X_T493[n, d0, d1, c : _T739, _T740, _T741, _T742] = =(X_T237[n, d0, d1, c])
// With Index Variables Made Integral:
// X_T493[n, d0, d1, c : _T739, _T740, _T741, _T742] = =(X_T237[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= c < 44, 0 <= c < 44, 0 <= d0 < 56, 0 <= d1 < 56, 0 <= d0 < 57, 0 <= d1 < 57, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= c < 44, 0 <= d0 < 56, 0 <= d1 < 56 }
// Defracted:
// X_T493[n, d0, d1, c : _T739, _T740, _T741, _T742] = =(X_T237[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range    X_T493    X_T237  
//        c        44         1         1  
//       d0        56      2508      2464  
//       d1        56        44        44  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 2464, 56 }
// Out stride: { 1, 2508 }
// Input 1 offset: 0
// Input 1 stride: { 1, 2464 }
// Tile size: { 32, 56 }
// Contraction output var shape: fp32(1, 57, 57, 44):(142956, 2508, 44, 1):558.422 KiB
// Computed true ops: 275968
// Computed work groups: 77
// Computed inner loops: 1
// Computed shared mem: 7296
// Computed out regs: 7168
// Computed mem read: 7168
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 19712, 1, 1
__kernel void kernel_c42_sdk_171(__global float* restrict  X_T493, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[7] = {0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[1824];
  int d1_c_gid = (get_group_id(0) * 32);
  {
    {
      int d1_c_tid = (tid % 32);
      int d0_tid = ((tid / 32) % 8);
      for (int d0_lid = 0; d0_lid < 7; d0_lid += 1)
      {
        int d0 = ((8 * d0_lid) + d0_tid);
        int lidx = ((57 * d1_c_tid) + d0);
        int gidx = ((d1_c_gid + d1_c_tid) + (2464 * d0));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)137983)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int d1_c_tid = (tid % 32);
    int d0_tid = ((tid / 32) % 8);
    for (int d0_lid = 0; d0_lid < 7; d0_lid += 1)
    {
      int d0 = ((8 * d0_lid) + d0_tid);
      float val1 = in1_shared[((57 * d1_c_tid) + d0)];
      agg[d0_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int d1_c_tid = (tid % 32);
  int d0_tid = ((tid / 32) % 8);
  for (int d0_lid = 0; d0_lid < 7; d0_lid += 1)
  {
    int d0 = ((8 * d0_lid) + d0_tid);
    float LX_T493 = agg[d0_lid];
    int gout_idx = ((d1_c_gid + d1_c_tid) + (2508 * d0));
    X_T493[gout_idx] = LX_T493;
  }
}
