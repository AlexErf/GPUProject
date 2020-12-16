#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 83 1
// lid: 256 1 1
// Original:
// X_T3041[n, d0, d1, c : _T4844, _T4845, _T4846, _T4847] = =(X_T2660[n, d0, d1, c])
// With Index Variables Made Integral:
// X_T3041[n, d0, d1, c : _T4844, _T4845, _T4846, _T4847] = =(X_T2660[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= d0 < 21, 0 <= d1 < 21, 0 <= d0 < 22, 0 <= d1 < 22, 0 <= c < 2016, 0 <= c < 2016, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= d0 < 21, 0 <= d1 < 21, 0 <= c < 2016 }
// Defracted:
// X_T3041[n, d0, d1, c : _T4844, _T4845, _T4846, _T4847] = =(X_T2660[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range   X_T3041   X_T2660  
//        c      2016         1         1  
//       d0        21     44352     42336  
//       d1        21      2016      2016  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 42336, 21 }
// Out stride: { 1, 44352 }
// Input 1 offset: 0
// Input 1 stride: { 1, 42336 }
// Tile size: { 512, 1 }
// Contraction output var shape: fp32(1, 22, 22, 2016):(975744, 44352, 2016, 1):3811.5 KiB
// Computed true ops: 1778112
// Computed work groups: 1743
// Computed inner loops: 1
// Computed shared mem: 2048
// Computed out regs: 2048
// Computed mem read: 2048
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 83, 1
__kernel void kernel_c42_sdk_1177(__global float* restrict  X_T3041, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[512];
  int d1_c_gid = (get_group_id(1) * 512);
  int d0_gid = get_group_id(0);
  {
    {
      int gbase = (d1_c_gid + (d0_gid * 42336));
      int d1_c_tid = (tid % 256);
      for (int d1_c_lid = 0; d1_c_lid < 2; d1_c_lid += 1)
      {
        int d1_c = ((256 * d1_c_lid) + d1_c_tid);
        int gidx = (gbase + d1_c);
        in1_shared[d1_c] = in1[clamp((int)gidx, (int)0, (int)889055)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int d1_c_tid = (tid % 256);
    for (int d1_c_lid = 0; d1_c_lid < 2; d1_c_lid += 1)
    {
      int d1_c = ((256 * d1_c_lid) + d1_c_tid);
      float val1 = in1_shared[d1_c];
      agg[d1_c_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int d1_c_tid = (tid % 256);
  for (int d1_c_lid = 0; d1_c_lid < 2; d1_c_lid += 1)
  {
    int d1_c_cond = ((d1_c_lid < 1) || ((d1_c_gid != 41984) || (d1_c_tid < 96)));
    if (d1_c_cond)
    {
      int d1_c = ((256 * d1_c_lid) + d1_c_tid);
      float LX_T3041 = agg[d1_c_lid];
      int gout_idx = ((d1_c_gid + d1_c) + (44352 * (int)d0_gid));
      X_T3041[gout_idx] = LX_T3041;
    }
  }
}
