#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 14 1
// lid: 256 1 1
// Original:
// X_T973[n0, n1, n2, a : _T1352, _T1353, _T1354, _T1355] = =(X_T972[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T973[n0, n1, n2, a : _T1352, _T1353, _T1354, _T1355] = =(X_T972[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 768, 0 <= a < 800, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 768 }
// Defracted:
// X_T973[n0, n1, n2, a : _T1352, _T1353, _T1354, _T1355] = =(X_T972[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T973    X_T972  
//        a       768         1         1  
//       n1        14     11200     10752  
//       n2        14       800       768  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 768, 14, 14 }
// Out stride: { 1, 11200, 800 }
// Input 1 offset: 0
// Input 1 stride: { 1, 10752, 768 }
// Tile size: { 128, 14, 1 }
// Contraction output var shape: fp32(1, 14, 14, 800):(156800, 11200, 800, 1):612.5 KiB
// Computed true ops: 301056
// Computed work groups: 84
// Computed inner loops: 1
// Computed shared mem: 7224
// Computed out regs: 8192
// Computed mem read: 7168
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 14, 1
__kernel void kernel_c108_sdk_324(__global float* restrict  X_T973, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[8] = {0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[1806];
  int a_gid = (get_group_id(0) * 128);
  int n2_gid = get_group_id(1);
  {
    {
      int gbase = (a_gid + (n2_gid * 768));
      int a_tid = (tid % 128);
      int n1_tid = ((tid / 128) % 2);
      for (int n1_lid = 0; n1_lid < 7; n1_lid += 1)
      {
        int n1 = ((2 * n1_lid) + n1_tid);
        int lidx = (a_tid + (129 * n1));
        int gidx = ((gbase + a_tid) + (10752 * n1));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)150527)];
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
        float LX_T973 = agg[(a_lid + (n1_lid * 4))];
        int gout_idx = (((a_gid + a) + (11200 * n1)) + (800 * n2_gid));
        X_T973[gout_idx] = LX_T973;
      }
    }
  }
}
