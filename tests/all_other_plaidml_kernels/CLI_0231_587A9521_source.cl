#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 7 1
// lid: 256 1 1
// Original:
// X_T2140[n, x0, x1, c : _T3391, _T3392, _T3393, _T3394] = >(X_T2139[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T2140[n, x0, x1, c : _T3391, _T3392, _T3393, _T3394] = >(X_T2139[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 7, 0 <= x1 < 7, 0 <= k0 + 2*x0 < 15, 0 <= k1 + 2*x1 < 15, 0 <= c < 176, 0 <= c < 176, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 7, 0 <= x1 < 7, 0 <= k0 + 2*x0 < 15, 0 <= k1 + 2*x1 < 15, 0 <= c < 176 }
// Defracted:
// X_T2140[n, x0, x1, c : _T3391, _T3392, _T3393, _T3394] = >(X_T2139[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T2140   X_T2139  
//        c       176         1         1  
//       k0         3         0      2640  
//       k1         3         0       176  
//       x0         7      1232      5280  
//       x1         7       176       352  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 176, 3, 3, 7, 7 }
// Out stride: { 1, 0, 0, 1232, 176 }
// Input 1 offset: 0
// Input 1 stride: { 1, 2640, 176, 5280, 352 }
// Tile size: { 64, 3, 3, 7, 1 }
// Contraction output var shape: fp32(1, 7, 7, 176):(8624, 1232, 176, 1):33.6875 KiB
// Computed true ops: 155232
// Computed work groups: 21
// Computed inner loops: 1
// Computed shared mem: 11520
// Computed out regs: 2048
// Computed mem read: 11520
// Computed mem write: 1792
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 7, 1
__kernel void kernel_c42_sdk_820(__global float* restrict  X_T2140, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[2] = {-FLT_MAX, -FLT_MAX, };
  __local float in1_shared[2880];
  int c_gid = (get_group_id(0) * 64);
  int x1_gid = get_group_id(1);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = (((c_gid + (k1_gid * 176)) + (x1_gid * 352)) + (k0_gid * 2640));
        int c_tid = (tid % 64);
        int k0_x0_tid = ((tid / 64) % 4);
        for (int k0_x0_lid = 0; k0_x0_lid < 4; k0_x0_lid += 1)
        {
          int k0_x0_cond = ((k0_x0_lid < 3) || (k0_x0_tid < 3));
          if (k0_x0_cond)
          {
            int k0_x0 = ((4 * k0_x0_lid) + k0_x0_tid);
            for (int k1_x1_lid = 0; k1_x1_lid < 3; k1_x1_lid += 1)
            {
              int lidx = (((45 * c_tid) + k0_x0) + (15 * k1_x1_lid));
              int gidx = (((gbase + c_tid) + (2640 * k0_x0)) + (176 * k1_x1_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)39599)];
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
      {
        for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
        {
          int c_tid = (tid % 32);
          int x0_tid = ((tid / 32) % 8);
          for (int c_lid = 0; c_lid < 2; c_lid += 1)
          {
            int c = ((32 * c_lid) + c_tid);
            int x0_cond = (x0_tid < 7);
            int x0 = select((int)0, (int)x0_tid, (int)x0_cond);
            float val1 = in1_shared[((((45 * c) + (15 * k1_lid)) + k0_lid) + (2 * x0))];
            float agg_rhs = select((float)agg[c_lid], (float)val1, (int)(val1 > agg[c_lid]));
            agg[c_lid] = select((float)agg[c_lid], (float)agg_rhs, (int)x0_cond);
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 32);
  int x0_tid = ((tid / 32) % 8);
  for (int c_lid = 0; c_lid < 2; c_lid += 1)
  {
    int c_cond = ((c_lid < 1) || ((c_gid != 128) || (c_tid < 16)));
    if (c_cond)
    {
      int c = ((32 * c_lid) + c_tid);
      int x0_cond = (x0_tid < 7);
      if (x0_cond)
      {
        float LX_T2140 = agg[c_lid];
        LX_T2140 = select((float)LX_T2140, (float)0, (int)(LX_T2140 == (float)-FLT_MAX));
        int gout_idx = (((c_gid + c) + (1232 * x0_tid)) + (176 * x1_gid));
        X_T2140[gout_idx] = LX_T2140;
      }
    }
  }
}
