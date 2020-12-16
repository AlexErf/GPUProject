#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2816 21 1
// lid: 256 1 1
// Original:
// X_T1780[n0, n1, n2, 168 + a : _T2826, _T2827, _T2828, _T2829] = =(X_T1779[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T1780[n0, n1, n2, 168 + a : _T2826, _T2827, _T2828, _T2829] = =(X_T1779[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 21, 0 <= n2 < 21, 0 <= n1 < 21, 0 <= n2 < 21, 0 <= a < 168, 0 <= 168 + a < 336, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 21, 0 <= n2 < 21, 0 <= a < 168 }
// Defracted:
// X_T1780[n0, n1, n2, 168 + a : _T2826, _T2827, _T2828, _T2829] = =(X_T1779[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T1780   X_T1779  
//        a       168         1         1  
//       n1        21      7056      3528  
//       n2        21       336       168  
//      off                 168         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 168, 21, 21 }
// Out stride: { 1, 7056, 336 }
// Input 1 offset: 0
// Input 1 stride: { 1, 3528, 168 }
// Tile size: { 168, 2, 1 }
// Contraction output var shape: fp32(1, 21, 21, 336):(148176, 7056, 336, 1):578.812 KiB
// Computed true ops: 148176
// Computed work groups: 231
// Computed inner loops: 1
// Computed shared mem: 1352
// Computed out regs: 2048
// Computed mem read: 1280
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2816, 21, 1
__kernel void kernel_c42_sdk_681(__global float* restrict  X_T1780, __global const float* restrict  in1)
{
  X_T1780 = (X_T1780 + 168);
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[338];
  int n2_gid = get_group_id(1);
  int n1_gid = (get_group_id(0) * 2);
  {
    {
      int gbase = ((n2_gid * 168) + (n1_gid * 3528));
      int a_n2_tid = (tid % 256);
      int a_n2_cond = (a_n2_tid < 168);
      if (a_n2_cond)
      {
        for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
        {
          int lidx = (a_n2_tid + (169 * n1_lid));
          int gidx = ((gbase + a_n2_tid) + (3528 * n1_lid));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)74087)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 128);
    int n1_tid = ((tid / 128) % 2);
    for (int a_lid = 0; a_lid < 2; a_lid += 1)
    {
      int a_cond = ((a_lid < 1) || (a_tid < 40));
      int a = select((int)0, (int)((128 * a_lid) + a_tid), (int)a_cond);
      float val1 = in1_shared[(a + (169 * n1_tid))];
      agg[a_lid] = select((float)agg[a_lid], (float)val1, (int)a_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 128);
  int n1_tid = ((tid / 128) % 2);
  int n1_cond = ((n1_gid != 20) || (n1_tid < 1));
  if (n1_cond)
  {
    for (int a_lid = 0; a_lid < 2; a_lid += 1)
    {
      int a_cond = ((a_lid < 1) || (a_tid < 40));
      if (a_cond)
      {
        int a = ((128 * a_lid) + a_tid);
        float LX_T1780 = agg[a_lid];
        int gout_idx = ((a + (7056 * (n1_gid + n1_tid))) + (336 * n2_gid));
        X_T1780[gout_idx] = LX_T1780;
      }
    }
  }
}
