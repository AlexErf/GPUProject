#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 9 1
// lid: 256 1 1
// Original:
// X_T831[n0, n1, n2, a : _T1130, _T1131, _T1132, _T1133] = =(X_T830[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T831[n0, n1, n2, a : _T1130, _T1131, _T1132, _T1133] = =(X_T830[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 576, 0 <= a < 608, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 576 }
// Defracted:
// X_T831[n0, n1, n2, a : _T1130, _T1131, _T1132, _T1133] = =(X_T830[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T831    X_T830  
//        a       576         1         1  
//       n1        14      8512      8064  
//       n2        14       608       576  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 576, 14, 14 }
// Out stride: { 1, 8512, 608 }
// Input 1 offset: 0
// Input 1 stride: { 1, 8064, 576 }
// Tile size: { 64, 2, 14 }
// Contraction output var shape: fp32(1, 14, 14, 608):(119168, 8512, 608, 1):465.5 KiB
// Computed true ops: 225792
// Computed work groups: 63
// Computed inner loops: 1
// Computed shared mem: 7280
// Computed out regs: 8192
// Computed mem read: 7168
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 9, 1
__kernel void kernel_c124_sdk_270(__global float* restrict  X_T831, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[8] = {0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[1820];
  int a_gid = (get_group_id(1) * 64);
  int n1_gid = (get_group_id(0) * 2);
  {
    {
      int gbase = (a_gid + (n1_gid * 8064));
      int a_tid = (tid % 64);
      int n2_n1_tid = ((tid / 64) % 4);
      for (int n2_n1_lid = 0; n2_n1_lid < 7; n2_n1_lid += 1)
      {
        int n2_n1 = ((4 * n2_n1_lid) + n2_n1_tid);
        int lidx = (a_tid + (65 * n2_n1));
        int gidx = ((gbase + a_tid) + (576 * n2_n1));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)112895)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 4);
    int n1_tid = ((tid / 128) % 2);
    for (int n2_lid = 0; n2_lid < 4; n2_lid += 1)
    {
      int n2_cond = ((n2_lid < 3) || (n2_tid < 2));
      int n2 = select((int)0, (int)((4 * n2_lid) + n2_tid), (int)n2_cond);
      for (int a_lid = 0; a_lid < 2; a_lid += 1)
      {
        int a = ((32 * a_lid) + a_tid);
        float val1 = in1_shared[((a + (65 * n2)) + (910 * n1_tid))];
        int agg_idx = (a_lid + (n2_lid * 2));
        agg[agg_idx] = select((float)agg[agg_idx], (float)val1, (int)n2_cond);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 4);
  int n1_tid = ((tid / 128) % 2);
  for (int n2_lid = 0; n2_lid < 4; n2_lid += 1)
  {
    int n2_cond = ((n2_lid < 3) || (n2_tid < 2));
    if (n2_cond)
    {
      int n2 = ((4 * n2_lid) + n2_tid);
      for (int a_lid = 0; a_lid < 2; a_lid += 1)
      {
        int a = ((32 * a_lid) + a_tid);
        float LX_T831 = agg[(a_lid + (n2_lid * 2))];
        int gout_idx = (((a_gid + a) + (8512 * (n1_gid + n1_tid))) + (608 * n2));
        X_T831[gout_idx] = LX_T831;
      }
    }
  }
}
