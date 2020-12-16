#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 7 1
// lid: 256 1 1
// Original:
// X_T1418[n0, n1, n2, a : _T2012, _T2013, _T2014, _T2015] = =(X_T1417[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T1418[n0, n1, n2, a : _T2012, _T2013, _T2014, _T2015] = =(X_T1417[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 672, 0 <= a < 704, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 672 }
// Defracted:
// X_T1418[n0, n1, n2, a : _T2012, _T2013, _T2014, _T2015] = =(X_T1417[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T1418   X_T1417  
//        a       672         1         1  
//       n1         7      4928      4704  
//       n2         7       704       672  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 672, 7, 7 }
// Out stride: { 1, 4928, 704 }
// Input 1 offset: 0
// Input 1 stride: { 1, 4704, 672 }
// Tile size: { 128, 1, 7 }
// Contraction output var shape: fp32(1, 7, 7, 704):(34496, 4928, 704, 1):134.75 KiB
// Computed true ops: 65856
// Computed work groups: 42
// Computed inner loops: 1
// Computed shared mem: 3584
// Computed out regs: 4096
// Computed mem read: 3584
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 7, 1
__kernel void kernel_c108_sdk_483(__global float* restrict  X_T1418, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[896];
  int a_gid = (get_group_id(0) * 128);
  int n1_gid = get_group_id(1);
  {
    {
      int gbase = (a_gid + (n1_gid * 4704));
      int a_tid = (tid % 128);
      int n2_n1_tid = ((tid / 128) % 2);
      for (int n2_n1_lid = 0; n2_n1_lid < 4; n2_n1_lid += 1)
      {
        int n2_n1_cond = ((n2_n1_lid < 3) || (n2_n1_tid < 1));
        if (n2_n1_cond)
        {
          int n2_n1 = ((2 * n2_n1_lid) + n2_n1_tid);
          int lidx = ((7 * a_tid) + n2_n1);
          int gidx = ((gbase + a_tid) + (672 * n2_n1));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)32927)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 8);
    for (int a_lid = 0; a_lid < 4; a_lid += 1)
    {
      int a = ((32 * a_lid) + a_tid);
      int n2_cond = (n2_tid < 7);
      int n2 = select((int)0, (int)n2_tid, (int)n2_cond);
      float val1 = in1_shared[((7 * a) + n2)];
      agg[a_lid] = select((float)agg[a_lid], (float)val1, (int)n2_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 8);
  for (int a_lid = 0; a_lid < 4; a_lid += 1)
  {
    int a_cond = ((a_lid < 1) || (a_gid != 640));
    if (a_cond)
    {
      int a = ((32 * a_lid) + a_tid);
      int n2_cond = (n2_tid < 7);
      if (n2_cond)
      {
        float LX_T1418 = agg[a_lid];
        int gout_idx = (((a_gid + a) + (4928 * n1_gid)) + (704 * n2_tid));
        X_T1418[gout_idx] = LX_T1418;
      }
    }
  }
}
