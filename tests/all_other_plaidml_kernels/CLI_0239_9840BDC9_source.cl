#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 7 1
// lid: 256 1 1
// Original:
// X_T2186[n0, n1, n2, a : _T3457, _T3458, _T3459, _T3460] = =(X_T2185[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T2186[n0, n1, n2, a : _T3457, _T3458, _T3459, _T3460] = =(X_T2185[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 176, 0 <= a < 704, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 176 }
// Defracted:
// X_T2186[n0, n1, n2, a : _T3457, _T3458, _T3459, _T3460] = =(X_T2185[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T2186   X_T2185  
//        a       176         1         1  
//       n1         7      4928      1232  
//       n2         7       704       176  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 176, 7, 7 }
// Out stride: { 1, 4928, 704 }
// Input 1 offset: 0
// Input 1 stride: { 1, 1232, 176 }
// Tile size: { 64, 7, 1 }
// Contraction output var shape: fp32(1, 7, 7, 704):(34496, 4928, 704, 1):134.75 KiB
// Computed true ops: 17248
// Computed work groups: 21
// Computed inner loops: 1
// Computed shared mem: 1792
// Computed out regs: 2048
// Computed mem read: 1792
// Computed mem write: 1792
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 7, 1
__kernel void kernel_c42_sdk_835(__global float* restrict  X_T2186, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[448];
  int a_gid = (get_group_id(0) * 64);
  int n2_gid = get_group_id(1);
  {
    {
      int gbase = (a_gid + (n2_gid * 176));
      int a_tid = (tid % 64);
      int n1_tid = ((tid / 64) % 4);
      for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
      {
        int n1_cond = ((n1_lid < 1) || (n1_tid < 3));
        if (n1_cond)
        {
          int n1 = ((4 * n1_lid) + n1_tid);
          int lidx = ((7 * a_tid) + n1);
          int gidx = ((gbase + a_tid) + (1232 * n1));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)8623)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n1_tid = ((tid / 32) % 8);
    for (int a_lid = 0; a_lid < 2; a_lid += 1)
    {
      int a = ((32 * a_lid) + a_tid);
      int n1_cond = (n1_tid < 7);
      int n1 = select((int)0, (int)n1_tid, (int)n1_cond);
      float val1 = in1_shared[((7 * a) + n1)];
      agg[a_lid] = select((float)agg[a_lid], (float)val1, (int)n1_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n1_tid = ((tid / 32) % 8);
  for (int a_lid = 0; a_lid < 2; a_lid += 1)
  {
    int a_cond = ((a_lid < 1) || ((a_gid != 128) || (a_tid < 16)));
    if (a_cond)
    {
      int a = ((32 * a_lid) + a_tid);
      int n1_cond = (n1_tid < 7);
      if (n1_cond)
      {
        float LX_T2186 = agg[a_lid];
        int gout_idx = (((a_gid + a) + (4928 * n1_tid)) + (704 * n2_gid));
        X_T2186[gout_idx] = LX_T2186;
      }
    }
  }
}
