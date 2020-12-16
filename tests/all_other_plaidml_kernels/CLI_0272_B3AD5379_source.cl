#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 11 11
// lid: 256 1 1
// Original:
// X_T3188[n0, n1, n2, 1344 + a : _T5073, _T5074, _T5075, _T5076] = =(X_T3187[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T3188[n0, n1, n2, 1344 + a : _T5073, _T5074, _T5075, _T5076] = =(X_T3187[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 11, 0 <= n2 < 11, 0 <= n1 < 11, 0 <= n2 < 11, 0 <= a < 672, 0 <= 1344 + a < 4032, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 11, 0 <= n2 < 11, 0 <= a < 672 }
// Defracted:
// X_T3188[n0, n1, n2, 1344 + a : _T5073, _T5074, _T5075, _T5076] = =(X_T3187[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T3188   X_T3187  
//        a       672         1         1  
//       n1        11     44352      7392  
//       n2        11      4032       672  
//      off                1344         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 672, 11, 11 }
// Out stride: { 1, 44352, 4032 }
// Input 1 offset: 0
// Input 1 stride: { 1, 7392, 672 }
// Tile size: { 256, 1, 1 }
// Contraction output var shape: fp32(1, 11, 11, 4032):(487872, 44352, 4032, 1):1905.75 KiB
// Computed true ops: 162624
// Computed work groups: 363
// Computed inner loops: 1
// Computed shared mem: 1024
// Computed out regs: 1024
// Computed mem read: 1024
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 11, 11
__kernel void kernel_c42_sdk_1233(__global float* restrict  X_T3188, __global const float* restrict  in1)
{
  X_T3188 = (X_T3188 + 1344);
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[256];
  int a_gid = (get_group_id(0) * 256);
  int n2_gid = get_group_id(1);
  int n1_gid = get_group_id(2);
  {
    {
      int gbase = ((a_gid + (n2_gid * 672)) + (n1_gid * 7392));
      int a_tid = (tid % 256);
      int gidx = (gbase + a_tid);
      in1_shared[a_tid] = in1[clamp((int)gidx, (int)0, (int)81311)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 256);
    float val1 = in1_shared[a_tid];
    agg[0] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 256);
  int a_cond = ((a_gid != 512) || (a_tid < 160));
  if (a_cond)
  {
    float LX_T3188 = agg[0];
    int gout_idx = (((a_gid + a_tid) + (44352 * (int)n1_gid)) + (4032 * n2_gid));
    X_T3188[gout_idx] = LX_T3188;
  }
}
