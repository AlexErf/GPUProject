#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Original:
// X_T2426[n0, n1, n2, a : _T3492, _T3493, _T3494, _T3495] = =(X_T2425[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T2426[n0, n1, n2, a : _T3492, _T3493, _T3494, _T3495] = =(X_T2425[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 1696, 0 <= a < 1728, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 1696 }
// Defracted:
// X_T2426[n0, n1, n2, a : _T3492, _T3493, _T3494, _T3495] = =(X_T2425[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T2426   X_T2425  
//        a      1696         1         1  
//       n1         7     12096     11872  
//       n2         7      1728      1696  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 1696, 7, 7 }
// Out stride: { 1, 12096, 1728 }
// Input 1 offset: 0
// Input 1 stride: { 1, 11872, 1696 }
// Tile size: { 1696, 1, 1 }
// Contraction output var shape: fp32(1, 7, 7, 1728):(84672, 12096, 1728, 1):330.75 KiB
// Computed true ops: 166208
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 6784
// Computed out regs: 7168
// Computed mem read: 6784
// Computed mem write: 6784
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_843(__global float* restrict  X_T2426, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[7] = {0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[1696];
  int n2_gid = get_group_id(0);
  int n1_gid = get_group_id(1);
  {
    {
      int gbase = ((n2_gid * 1696) + (n1_gid * 11872));
      int a_n2_tid = (tid % 256);
      for (int a_n2_lid = 0; a_n2_lid < 7; a_n2_lid += 1)
      {
        int a_n2_cond = ((a_n2_lid < 6) || (a_n2_tid < 160));
        if (a_n2_cond)
        {
          int a_n2 = ((256 * a_n2_lid) + a_n2_tid);
          int gidx = (gbase + a_n2);
          in1_shared[a_n2] = in1[clamp((int)gidx, (int)0, (int)83103)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 256);
    for (int a_lid = 0; a_lid < 7; a_lid += 1)
    {
      int a_cond = ((a_lid < 6) || (a_tid < 160));
      int a = select((int)0, (int)((256 * a_lid) + a_tid), (int)a_cond);
      float val1 = in1_shared[a];
      agg[a_lid] = select((float)agg[a_lid], (float)val1, (int)a_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 256);
  for (int a_lid = 0; a_lid < 7; a_lid += 1)
  {
    int a_cond = ((a_lid < 6) || (a_tid < 160));
    if (a_cond)
    {
      int a = ((256 * a_lid) + a_tid);
      float LX_T2426 = agg[a_lid];
      int gout_idx = ((a + (12096 * n1_gid)) + (1728 * n2_gid));
      X_T2426[gout_idx] = LX_T2426;
    }
  }
}
