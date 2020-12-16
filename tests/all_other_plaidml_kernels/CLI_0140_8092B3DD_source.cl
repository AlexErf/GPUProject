#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 17 1
// lid: 256 1 1
// Original:
// X_T931[n0, n1, n2, 768 + a : _T1286, _T1287, _T1288, _T1289] = =(X_T930[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T931[n0, n1, n2, 768 + a : _T1286, _T1287, _T1288, _T1289] = =(X_T930[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 17, 0 <= n2 < 17, 0 <= n1 < 17, 0 <= n2 < 17, 0 <= a < 320, 0 <= 768 + a < 1088, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 17, 0 <= n2 < 17, 0 <= a < 320 }
// Defracted:
// X_T931[n0, n1, n2, 768 + a : _T1286, _T1287, _T1288, _T1289] = =(X_T930[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T931    X_T930  
//        a       320         1         1  
//       n1        17     18496      5440  
//       n2        17      1088       320  
//      off                 768         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 320, 17, 17 }
// Out stride: { 1, 18496, 1088 }
// Input 1 offset: 0
// Input 1 stride: { 1, 5440, 320 }
// Tile size: { 64, 1, 17 }
// Contraction output var shape: fp32(1, 17, 17, 1088):(314432, 18496, 1088, 1):1228.25 KiB
// Computed true ops: 184960
// Computed work groups: 85
// Computed inner loops: 1
// Computed shared mem: 4352
// Computed out regs: 6144
// Computed mem read: 4352
// Computed mem write: 4352
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1280, 17, 1
__kernel void kernel_c51_sdk_304(__global float* restrict  X_T931, __global const float* restrict  in1)
{
  X_T931 = (X_T931 + 768);
  int tid = get_local_id(0);
  float agg[6] = {0, 0, 0, 0, 0, 0, };
  __local float in1_shared[1088];
  int a_gid = (get_group_id(0) * 64);
  int n1_gid = get_group_id(1);
  {
    {
      int gbase = (a_gid + (n1_gid * 5440));
      int a_tid = (tid % 64);
      int n2_n1_tid = ((tid / 64) % 4);
      for (int n2_n1_lid = 0; n2_n1_lid < 5; n2_n1_lid += 1)
      {
        int n2_n1_cond = ((n2_n1_lid < 4) || (n2_n1_tid < 1));
        if (n2_n1_cond)
        {
          int n2_n1 = ((4 * n2_n1_lid) + n2_n1_tid);
          int lidx = ((17 * a_tid) + n2_n1);
          int gidx = ((gbase + a_tid) + (320 * n2_n1));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)92479)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 8);
    for (int n2_lid = 0; n2_lid < 3; n2_lid += 1)
    {
      int n2_cond = ((n2_lid < 2) || (n2_tid < 1));
      int n2 = select((int)0, (int)((8 * n2_lid) + n2_tid), (int)n2_cond);
      for (int a_lid = 0; a_lid < 2; a_lid += 1)
      {
        int a = ((32 * a_lid) + a_tid);
        float val1 = in1_shared[((17 * a) + n2)];
        int agg_idx = (a_lid + (n2_lid * 2));
        agg[agg_idx] = select((float)agg[agg_idx], (float)val1, (int)n2_cond);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 8);
  for (int n2_lid = 0; n2_lid < 3; n2_lid += 1)
  {
    int n2_cond = ((n2_lid < 2) || (n2_tid < 1));
    if (n2_cond)
    {
      int n2 = ((8 * n2_lid) + n2_tid);
      for (int a_lid = 0; a_lid < 2; a_lid += 1)
      {
        int a = ((32 * a_lid) + a_tid);
        float LX_T931 = agg[(a_lid + (n2_lid * 2))];
        int gout_idx = (((a_gid + a) + (18496 * n1_gid)) + (1088 * n2));
        X_T931[gout_idx] = LX_T931;
      }
    }
  }
}
