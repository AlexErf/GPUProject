#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Original:
// X_T1248[n0, n1, n2, a : _T1790, _T1791, _T1792, _T1793] = =(X_T1247[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T1248[n0, n1, n2, a : _T1790, _T1791, _T1792, _T1793] = =(X_T1247[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 608, 0 <= a < 640, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 608 }
// Defracted:
// X_T1248[n0, n1, n2, a : _T1790, _T1791, _T1792, _T1793] = =(X_T1247[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T1248   X_T1247  
//        a       608         1         1  
//       n1         7      4480      4256  
//       n2         7       640       608  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 608, 7, 7 }
// Out stride: { 1, 4480, 640 }
// Input 1 offset: 0
// Input 1 stride: { 1, 4256, 608 }
// Tile size: { 608, 1, 1 }
// Contraction output var shape: fp32(1, 7, 7, 640):(31360, 4480, 640, 1):122.5 KiB
// Computed true ops: 59584
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 2432
// Computed out regs: 3072
// Computed mem read: 2432
// Computed mem write: 2432
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c68_sdk_429(__global float* restrict  X_T1248, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[3] = {0, 0, 0, };
  __local float in1_shared[608];
  int n2_gid = get_group_id(0);
  int n1_gid = get_group_id(1);
  {
    {
      int gbase = ((n2_gid * 608) + (n1_gid * 4256));
      int a_n2_tid = (tid % 256);
      for (int a_n2_lid = 0; a_n2_lid < 3; a_n2_lid += 1)
      {
        int a_n2_cond = ((a_n2_lid < 2) || (a_n2_tid < 96));
        if (a_n2_cond)
        {
          int a_n2 = ((256 * a_n2_lid) + a_n2_tid);
          int gidx = (gbase + a_n2);
          in1_shared[a_n2] = in1[clamp((int)gidx, (int)0, (int)29791)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 256);
    for (int a_lid = 0; a_lid < 3; a_lid += 1)
    {
      int a_cond = ((a_lid < 2) || (a_tid < 96));
      int a = select((int)0, (int)((256 * a_lid) + a_tid), (int)a_cond);
      float val1 = in1_shared[a];
      agg[a_lid] = select((float)agg[a_lid], (float)val1, (int)a_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 256);
  for (int a_lid = 0; a_lid < 3; a_lid += 1)
  {
    int a_cond = ((a_lid < 2) || (a_tid < 96));
    if (a_cond)
    {
      int a = ((256 * a_lid) + a_tid);
      float LX_T1248 = agg[a_lid];
      int gout_idx = ((a + (4480 * n1_gid)) + (640 * n2_gid));
      X_T1248[gout_idx] = LX_T1248;
    }
  }
}
