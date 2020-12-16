#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2304 17 1
// lid: 256 1 1
// Original:
// X_T980[n0, n1, n2, 192 + a : _T1348, _T1349, _T1350, _T1351] = =(X_T979[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T980[n0, n1, n2, 192 + a : _T1348, _T1349, _T1350, _T1351] = =(X_T979[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 17, 0 <= n2 < 17, 0 <= n1 < 17, 0 <= n2 < 17, 0 <= a < 192, 0 <= 192 + a < 384, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 17, 0 <= n2 < 17, 0 <= a < 192 }
// Defracted:
// X_T980[n0, n1, n2, 192 + a : _T1348, _T1349, _T1350, _T1351] = =(X_T979[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T980    X_T979  
//        a       192         1         1  
//       n1        17      6528      3264  
//       n2        17       384       192  
//      off                 192         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 192, 17, 17 }
// Out stride: { 1, 6528, 384 }
// Input 1 offset: 0
// Input 1 stride: { 1, 3264, 192 }
// Tile size: { 192, 2, 1 }
// Contraction output var shape: fp32(1, 17, 17, 384):(110976, 6528, 384, 1):433.5 KiB
// Computed true ops: 110976
// Computed work groups: 153
// Computed inner loops: 1
// Computed shared mem: 1544
// Computed out regs: 2048
// Computed mem read: 1536
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2304, 17, 1
__kernel void kernel_c51_sdk_319(__global float* restrict  X_T980, __global const float* restrict  in1)
{
  X_T980 = (X_T980 + 192);
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[386];
  int n2_gid = get_group_id(1);
  int n1_gid = (get_group_id(0) * 2);
  {
    {
      int gbase = ((n2_gid * 192) + (n1_gid * 3264));
      int a_n2_tid = (tid % 256);
      int a_n2_cond = (a_n2_tid < 192);
      if (a_n2_cond)
      {
        for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
        {
          int lidx = (a_n2_tid + (193 * n1_lid));
          int gidx = ((gbase + a_n2_tid) + (3264 * n1_lid));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)55487)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 128);
    int n1_tid = ((tid / 128) % 2);
    for (int a_lid = 0; a_lid < 2; a_lid += 1)
    {
      int a_cond = ((a_lid < 1) || (a_tid < 64));
      int a = select((int)0, (int)((128 * a_lid) + a_tid), (int)a_cond);
      float val1 = in1_shared[(a + (193 * n1_tid))];
      agg[a_lid] = select((float)agg[a_lid], (float)val1, (int)a_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 128);
  int n1_tid = ((tid / 128) % 2);
  int n1_cond = ((n1_gid != 16) || (n1_tid < 1));
  if (n1_cond)
  {
    for (int a_lid = 0; a_lid < 2; a_lid += 1)
    {
      int a_cond = ((a_lid < 1) || (a_tid < 64));
      if (a_cond)
      {
        int a = ((128 * a_lid) + a_tid);
        float LX_T980 = agg[a_lid];
        int gout_idx = ((a + (6528 * (n1_gid + n1_tid))) + (384 * n2_gid));
        X_T980[gout_idx] = LX_T980;
      }
    }
  }
}
