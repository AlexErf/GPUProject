#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 8 1
// lid: 256 1 1
// Original:
// X_T228[n, x0, x1, g, gco : _T234, _T235, _T236, _T237, _T238] = +(X_T227[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T41[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T228[n, x0, x1, g, gco : _T234, _T235, _T236, _T237, _T238] = +(X_T227[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T41[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + 2*x0 < 29, 0 <= k1 + 2*x1 < 29, 0 <= g < 256, 0 <= g + gci < 256, 0 <= g < 256, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + 2*x0 < 29, 0 <= k1 + 2*x1 < 29, 0 <= g < 256, 0 <= g + gci < 256 }
// Defracted:
// X_T228[n, x0, x1, g, gco : _T234, _T235, _T236, _T237, _T238] = +(X_T227[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T41[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T228    X_T227     X_T41  
//        g       256         1         1         1  
//       k0         3         0      7424       768  
//       k1         3         0       256       256  
//       x0        14      3584     14848         0  
//       x1        14       256       512         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 256, 3, 3, 14, 14 }
// Out stride: { 1, 0, 0, 3584, 256 }
// Input 1 offset: 0
// Input 1 stride: { 1, 7424, 256, 14848, 512 }
// Input 2 offset: 0
// Input 2 stride: { 1, 768, 256, 0, 0 }
// Tile size: { 32, 3, 3, 14, 2 }
// Contraction output var shape: fp32(1, 14, 14, 256, 1):(50176, 3584, 256, 1, 1):196 KiB
// Computed true ops: 903168
// Computed work groups: 56
// Computed inner loops: 1
// Computed shared mem: 19712
// Computed out regs: 4096
// Computed mem read: 19712
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 8, 1
__kernel void kernel_c25_sdk_56(__global float* restrict  X_T228, __global const float* restrict  in1, __global const float* restrict  in2)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[4640];
  __local float in2_shared[288];
  int g_gid = (get_group_id(1) * 32);
  int x1_gid = (get_group_id(0) * 2);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = (((g_gid + (k1_gid * 256)) + (x1_gid * 512)) + (k0_gid * 7424));
        int g_tid = (tid % 32);
        int k0_x0_tid = ((tid / 32) % 8);
        for (int k0_x0_lid = 0; k0_x0_lid < 4; k0_x0_lid += 1)
        {
          int k0_x0_cond = ((k0_x0_lid < 3) || (k0_x0_tid < 5));
          if (k0_x0_cond)
          {
            int k0_x0 = ((8 * k0_x0_lid) + k0_x0_tid);
            for (int k1_x1_lid = 0; k1_x1_lid < 5; k1_x1_lid += 1)
            {
              int lidx = (((145 * g_tid) + k0_x0) + (29 * k1_x1_lid));
              int gidx = (((gbase + g_tid) + (7424 * k0_x0)) + (256 * k1_x1_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)215295)];
            }
          }
        }
      }
      {
        int gbase = ((g_gid + (k1_gid * 256)) + (k0_gid * 768));
        int g_tid = (tid % 32);
        int k1_k0_tid = ((tid / 32) % 8);
        for (int k1_k0_lid = 0; k1_k0_lid < 2; k1_k0_lid += 1)
        {
          int k1_k0_cond = ((k1_k0_lid < 1) || (k1_k0_tid < 1));
          if (k1_k0_cond)
          {
            int k1_k0 = ((8 * k1_k0_lid) + k1_k0_tid);
            int lidx = ((9 * g_tid) + k1_k0);
            int gidx = ((gbase + g_tid) + (256 * k1_k0));
            in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)2303)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
      {
        for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
        {
          int g_tid = (tid % 32);
          int x1_tid = ((tid / 32) % 2);
          int x0_tid = ((tid / 64) % 4);
          for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
          {
            int x0_cond = ((x0_lid < 3) || (x0_tid < 2));
            int x0 = select((int)0, (int)((4 * x0_lid) + x0_tid), (int)x0_cond);
            float val1 = in1_shared[(((((145 * g_tid) + (29 * k1_lid)) + (58 * x1_tid)) + k0_lid) + (2 * x0))];
            float val2 = in2_shared[(((9 * g_tid) + k1_lid) + (3 * k0_lid))];
            float agg_rhs = mad(val2, val1, agg[x0_lid]);
            agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)x0_cond);
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int g_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 2);
  int x0_tid = ((tid / 64) % 4);
  for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
  {
    int x0_cond = ((x0_lid < 3) || (x0_tid < 2));
    if (x0_cond)
    {
      int x0 = ((4 * x0_lid) + x0_tid);
      float LX_T228 = agg[x0_lid];
      int gout_idx = (((g_gid + g_tid) + (3584 * x0)) + (256 * (x1_gid + x1_tid)));
      X_T228[gout_idx] = LX_T228;
    }
  }
}
