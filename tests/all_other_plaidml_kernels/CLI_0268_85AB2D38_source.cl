#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2816 11 1
// lid: 256 1 1
// Original:
// X_T3048[n0, n1, n2, 336 + a : _T4873, _T4874, _T4875, _T4876] = =(X_T3047[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T3048[n0, n1, n2, 336 + a : _T4873, _T4874, _T4875, _T4876] = =(X_T3047[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 11, 0 <= n2 < 11, 0 <= n1 < 11, 0 <= n2 < 11, 0 <= a < 336, 0 <= 336 + a < 672, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 11, 0 <= n2 < 11, 0 <= a < 336 }
// Defracted:
// X_T3048[n0, n1, n2, 336 + a : _T4873, _T4874, _T4875, _T4876] = =(X_T3047[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T3048   X_T3047  
//        a       336         1         1  
//       n1        11      7392      3696  
//       n2        11       672       336  
//      off                 336         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 336, 11, 11 }
// Out stride: { 1, 7392, 672 }
// Input 1 offset: 0
// Input 1 stride: { 1, 3696, 336 }
// Tile size: { 336, 1, 1 }
// Contraction output var shape: fp32(1, 11, 11, 672):(81312, 7392, 672, 1):317.625 KiB
// Computed true ops: 81312
// Computed work groups: 121
// Computed inner loops: 1
// Computed shared mem: 1344
// Computed out regs: 2048
// Computed mem read: 1280
// Computed mem write: 1408
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2816, 11, 1
__kernel void kernel_c42_sdk_1183(__global float* restrict  X_T3048, __global const float* restrict  in1)
{
  X_T3048 = (X_T3048 + 336);
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[336];
  int n2_gid = get_group_id(0);
  int n1_gid = get_group_id(1);
  {
    {
      int gbase = ((n2_gid * 336) + (n1_gid * 3696));
      int a_n2_tid = (tid % 256);
      for (int a_n2_lid = 0; a_n2_lid < 2; a_n2_lid += 1)
      {
        int a_n2_cond = ((a_n2_lid < 1) || (a_n2_tid < 80));
        if (a_n2_cond)
        {
          int a_n2 = ((256 * a_n2_lid) + a_n2_tid);
          int gidx = (gbase + a_n2);
          in1_shared[a_n2] = in1[clamp((int)gidx, (int)0, (int)40655)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 256);
    for (int a_lid = 0; a_lid < 2; a_lid += 1)
    {
      int a_cond = ((a_lid < 1) || (a_tid < 80));
      int a = select((int)0, (int)((256 * a_lid) + a_tid), (int)a_cond);
      float val1 = in1_shared[a];
      agg[a_lid] = select((float)agg[a_lid], (float)val1, (int)a_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 256);
  for (int a_lid = 0; a_lid < 2; a_lid += 1)
  {
    int a_cond = ((a_lid < 1) || (a_tid < 80));
    if (a_cond)
    {
      int a = ((256 * a_lid) + a_tid);
      float LX_T3048 = agg[a_lid];
      int gout_idx = ((a + (7392 * n1_gid)) + (672 * n2_gid));
      X_T3048[gout_idx] = LX_T3048;
    }
  }
}
