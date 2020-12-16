#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2560 14 1
// lid: 256 1 1
// Original:
// X_T1381[n0, n1, n2, a : _T1944, _T1945, _T1946, _T1947] = =(X_T1380[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T1381[n0, n1, n2, a : _T1944, _T1945, _T1946, _T1947] = =(X_T1380[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 1280, 0 <= a < 1312, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 1280 }
// Defracted:
// X_T1381[n0, n1, n2, a : _T1944, _T1945, _T1946, _T1947] = =(X_T1380[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T1381   X_T1380  
//        a      1280         1         1  
//       n1        14     18368     17920  
//       n2        14      1312      1280  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 1280, 14, 14 }
// Out stride: { 1, 18368, 1312 }
// Input 1 offset: 0
// Input 1 stride: { 1, 17920, 1280 }
// Tile size: { 128, 14, 1 }
// Contraction output var shape: fp32(1, 14, 14, 1312):(257152, 18368, 1312, 1):1004.5 KiB
// Computed true ops: 501760
// Computed work groups: 140
// Computed inner loops: 1
// Computed shared mem: 7224
// Computed out regs: 8192
// Computed mem read: 7168
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2560, 14, 1
__kernel void kernel_c124_sdk_468(__global float* restrict  X_T1381, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[8] = {0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[1806];
  int a_gid = (get_group_id(0) * 128);
  int n2_gid = get_group_id(1);
  {
    {
      int gbase = (a_gid + (n2_gid * 1280));
      int a_tid = (tid % 128);
      int n1_tid = ((tid / 128) % 2);
      for (int n1_lid = 0; n1_lid < 7; n1_lid += 1)
      {
        int n1 = ((2 * n1_lid) + n1_tid);
        int lidx = (a_tid + (129 * n1));
        int gidx = ((gbase + a_tid) + (17920 * n1));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)250879)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n1_tid = ((tid / 32) % 8);
    for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
    {
      int n1_cond = ((n1_lid < 1) || (n1_tid < 6));
      int n1 = select((int)0, (int)((8 * n1_lid) + n1_tid), (int)n1_cond);
      for (int a_lid = 0; a_lid < 4; a_lid += 1)
      {
        int a = ((32 * a_lid) + a_tid);
        float val1 = in1_shared[(a + (129 * n1))];
        int agg_idx = (a_lid + (n1_lid * 4));
        agg[agg_idx] = select((float)agg[agg_idx], (float)val1, (int)n1_cond);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n1_tid = ((tid / 32) % 8);
  for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
  {
    int n1_cond = ((n1_lid < 1) || (n1_tid < 6));
    if (n1_cond)
    {
      int n1 = ((8 * n1_lid) + n1_tid);
      for (int a_lid = 0; a_lid < 4; a_lid += 1)
      {
        int a = ((32 * a_lid) + a_tid);
        float LX_T1381 = agg[(a_lid + (n1_lid * 4))];
        int gout_idx = (((a_gid + a) + (18368 * n1)) + (1312 * n2_gid));
        X_T1381[gout_idx] = LX_T1381;
      }
    }
  }
}
