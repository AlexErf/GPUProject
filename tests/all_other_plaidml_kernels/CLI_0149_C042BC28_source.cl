#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 14080 83 1
// lid: 256 1 1
// Original:
// X_T492[n, d0, d1, c : _T739, _T740, _T741, _T742] = =(X_T240[n, d0, d1, c])
// With Index Variables Made Integral:
// X_T492[n, d0, d1, c : _T739, _T740, _T741, _T742] = =(X_T240[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= d0 < 83, 0 <= d1 < 83, 0 <= d0 < 84, 0 <= d1 < 84, 0 <= c < 168, 0 <= c < 168, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= d0 < 83, 0 <= d1 < 83, 0 <= c < 168 }
// Defracted:
// X_T492[n, d0, d1, c : _T739, _T740, _T741, _T742] = =(X_T240[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range    X_T492    X_T240  
//        c       168         1         1  
//       d0        83     14112     13944  
//       d1        83       168       168  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 13944, 83 }
// Out stride: { 1, 14112 }
// Input 1 offset: 0
// Input 1 stride: { 1, 13944 }
// Tile size: { 256, 1 }
// Contraction output var shape: fp32(1, 84, 84, 168):(1185408, 14112, 168, 1):4630.5 KiB
// Computed true ops: 2314704
// Computed work groups: 4565
// Computed inner loops: 1
// Computed shared mem: 1024
// Computed out regs: 1024
// Computed mem read: 1024
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 14080, 83, 1
__kernel void kernel_c42_sdk_171(__global float* restrict  X_T492, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[256];
  int d1_c_gid = (get_group_id(0) * 256);
  int d0_gid = get_group_id(1);
  {
    {
      int gbase = (d1_c_gid + (d0_gid * 13944));
      int d1_c_tid = (tid % 256);
      int gidx = (gbase + d1_c_tid);
      in1_shared[d1_c_tid] = in1[clamp((int)gidx, (int)0, (int)1157351)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int d1_c_tid = (tid % 256);
    float val1 = in1_shared[d1_c_tid];
    agg[0] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int d1_c_tid = (tid % 256);
  int d1_c_cond = ((d1_c_gid != 13824) || (d1_c_tid < 120));
  if (d1_c_cond)
  {
    float LX_T492 = agg[0];
    int gout_idx = ((d1_c_gid + d1_c_tid) + (14112 * d0_gid));
    X_T492[gout_idx] = LX_T492;
  }
}
