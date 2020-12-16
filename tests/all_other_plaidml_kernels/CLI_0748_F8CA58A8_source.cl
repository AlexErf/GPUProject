#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Original:
// X_T2551[n0, n1, n2, a : _T3677, _T3678, _T3679, _T3680] = =(X_T2550[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T2551[n0, n1, n2, a : _T3677, _T3678, _T3679, _T3680] = =(X_T2550[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 1856, 0 <= a < 1888, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 1856 }
// Defracted:
// X_T2551[n0, n1, n2, a : _T3677, _T3678, _T3679, _T3680] = =(X_T2550[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T2551   X_T2550  
//        a      1856         1         1  
//       n1         7     13216     12992  
//       n2         7      1888      1856  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 1856, 7, 7 }
// Out stride: { 1, 13216, 1888 }
// Input 1 offset: 0
// Input 1 stride: { 1, 12992, 1856 }
// Tile size: { 1856, 1, 1 }
// Contraction output var shape: fp32(1, 7, 7, 1888):(92512, 13216, 1888, 1):361.375 KiB
// Computed true ops: 181888
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 7424
// Computed out regs: 8192
// Computed mem read: 7424
// Computed mem write: 7424
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_888(__global float* restrict  X_T2551, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[8] = {0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[1856];
  int n2_gid = get_group_id(0);
  int n1_gid = get_group_id(1);
  {
    {
      int gbase = ((n2_gid * 1856) + (n1_gid * 12992));
      int a_n2_tid = (tid % 256);
      for (int a_n2_lid = 0; a_n2_lid < 8; a_n2_lid += 1)
      {
        int a_n2_cond = ((a_n2_lid < 7) || (a_n2_tid < 64));
        if (a_n2_cond)
        {
          int a_n2 = ((256 * a_n2_lid) + a_n2_tid);
          int gidx = (gbase + a_n2);
          in1_shared[a_n2] = in1[clamp((int)gidx, (int)0, (int)90943)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 256);
    for (int a_lid = 0; a_lid < 8; a_lid += 1)
    {
      int a_cond = ((a_lid < 7) || (a_tid < 64));
      int a = select((int)0, (int)((256 * a_lid) + a_tid), (int)a_cond);
      float val1 = in1_shared[a];
      agg[a_lid] = select((float)agg[a_lid], (float)val1, (int)a_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 256);
  for (int a_lid = 0; a_lid < 8; a_lid += 1)
  {
    int a_cond = ((a_lid < 7) || (a_tid < 64));
    if (a_cond)
    {
      int a = ((256 * a_lid) + a_tid);
      float LX_T2551 = agg[a_lid];
      int gout_idx = ((a + (13216 * n1_gid)) + (1888 * n2_gid));
      X_T2551[gout_idx] = LX_T2551;
    }
  }
}