#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Original:
// X_T2451[n0, n1, n2, a : _T3529, _T3530, _T3531, _T3532] = =(X_T2450[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T2451[n0, n1, n2, a : _T3529, _T3530, _T3531, _T3532] = =(X_T2450[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 1728, 0 <= a < 1760, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 1728 }
// Defracted:
// X_T2451[n0, n1, n2, a : _T3529, _T3530, _T3531, _T3532] = =(X_T2450[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T2451   X_T2450  
//        a      1728         1         1  
//       n1         7     12320     12096  
//       n2         7      1760      1728  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 1728, 7, 7 }
// Out stride: { 1, 12320, 1760 }
// Input 1 offset: 0
// Input 1 stride: { 1, 12096, 1728 }
// Tile size: { 1728, 1, 1 }
// Contraction output var shape: fp32(1, 7, 7, 1760):(86240, 12320, 1760, 1):336.875 KiB
// Computed true ops: 169344
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 6912
// Computed out regs: 7168
// Computed mem read: 6912
// Computed mem write: 6912
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_852(__global float* restrict  X_T2451, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[7] = {0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[1728];
  int n2_gid = get_group_id(0);
  int n1_gid = get_group_id(1);
  {
    {
      int gbase = ((n2_gid * 1728) + (n1_gid * 12096));
      int a_n2_tid = (tid % 256);
      for (int a_n2_lid = 0; a_n2_lid < 7; a_n2_lid += 1)
      {
        int a_n2_cond = ((a_n2_lid < 6) || (a_n2_tid < 192));
        if (a_n2_cond)
        {
          int a_n2 = ((256 * a_n2_lid) + a_n2_tid);
          int gidx = (gbase + a_n2);
          in1_shared[a_n2] = in1[clamp((int)gidx, (int)0, (int)84671)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 256);
    for (int a_lid = 0; a_lid < 7; a_lid += 1)
    {
      int a_cond = ((a_lid < 6) || (a_tid < 192));
      int a = select((int)0, (int)((256 * a_lid) + a_tid), (int)a_cond);
      float val1 = in1_shared[a];
      agg[a_lid] = select((float)agg[a_lid], (float)val1, (int)a_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 256);
  for (int a_lid = 0; a_lid < 7; a_lid += 1)
  {
    int a_cond = ((a_lid < 6) || (a_tid < 192));
    if (a_cond)
    {
      int a = ((256 * a_lid) + a_tid);
      float LX_T2451 = agg[a_lid];
      int gout_idx = ((a + (12320 * n1_gid)) + (1760 * n2_gid));
      X_T2451[gout_idx] = LX_T2451;
    }
  }
}
