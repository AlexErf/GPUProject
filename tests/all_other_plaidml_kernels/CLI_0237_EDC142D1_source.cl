#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 2 1
// lid: 256 1 1
// Original:
// X_T2174[n, x0, x1, g, gco : _T3438, _T3439, _T3440, _T3441, _T3442] = +(X_T2173[n, -3 + k0 + x0, -3 + k1 + x1, g + gci] * X_T2146[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T2174[n, x0, x1, g, gco : _T3438, _T3439, _T3440, _T3441, _T3442] = +(X_T2173[n, -3 + k0 + x0, -3 + k1 + x1, g + gci] * X_T2146[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= x0 < 7, 0 <= x1 < 7, 0 <= -3 + k0 + x0 < 7, 0 <= -3 + k1 + x1 < 7, 0 <= k0 < 7, 0 <= k1 < 7, 0 <= g < 176, 0 <= g + gci < 176, 0 <= g < 176, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= x0 < 7, 0 <= x1 < 7, 0 <= -3 + k0 + x0 < 7, 0 <= -3 + k1 + x1 < 7, 0 <= k0 < 7, 0 <= k1 < 7, 0 <= g < 176, 0 <= g + gci < 176 }
// Defracted:
// X_T2174[n, x0, x1, g, gco : _T3438, _T3439, _T3440, _T3441, _T3442] = +(X_T2173[n, -3 + k0 + x0, -3 + k1 + x1, g + gci] * X_T2146[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T2174   X_T2173   X_T2146  
//        g       176         1         1         1  
//       k0         7         0      1232      1232  
//       k1         7         0       176       176  
//       x0         7      1232      1232         0  
//       x1         7       176       176         0  
//      off                   0     -4224         0  
//      vec                   1         1         1  
// Constraint: (0,-1,0,-1,0) <= -3
// Constraint: (0,1,0,1,0) <= 9
// Constraint: (0,0,-1,0,-1) <= -3
// Constraint: (0,0,1,0,1) <= 9
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 176, 7, 7, 7, 7 }
// Out stride: { 1, 0, 0, 1232, 176 }
// Input 1 offset: -4224
// Input 1 stride: { 1, 1232, 176, 1232, 176 }
// Input 2 offset: 0
// Input 2 stride: { 1, 1232, 176, 0, 0 }
// Tile size: { 32, 7, 7, 7, 4 }
// Contraction output var shape: fp32(1, 7, 7, 176, 1):(8624, 1232, 176, 1, 1):33.6875 KiB
// Computed true ops: 845152
// Computed work groups: 12
// Computed inner loops: 1
// Computed shared mem: 18432
// Computed out regs: 4096
// Computed mem read: 18304
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 2, 1
__kernel void kernel_c42_sdk_831(__global float* restrict  X_T2174, __global const float* restrict  in1, __global const float* restrict  in2)
{
  in1 = (in1 + -4224);
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[3040];
  __local float in2_shared[1568];
  int g_gid = (get_group_id(0) * 32);
  int x1_gid = (get_group_id(1) * 4);
  for (int k0_gid = 0; k0_gid < 7; k0_gid += 7)
  {
    for (int k1_gid = 0; k1_gid < 7; k1_gid += 7)
    {
      {
        int gbase = (((g_gid + (k1_gid * 176)) + (x1_gid * 176)) + (k0_gid * 1232));
        int g_tid = (tid % 32);
        int k1_x1_k0_x0_tid = ((tid / 32) % 8);
        for (int k1_x1_k0_x0_lid = 0; k1_x1_k0_x0_lid < 12; k1_x1_k0_x0_lid += 1)
        {
          int k1_x1_k0_x0_cond = ((k1_x1_k0_x0_lid < 11) || (k1_x1_k0_x0_tid < 6));
          if (k1_x1_k0_x0_cond)
          {
            int k1_x1_k0_x0 = ((8 * k1_x1_k0_x0_lid) + k1_x1_k0_x0_tid);
            int lidx = ((95 * g_tid) + k1_x1_k0_x0);
            int gidx = ((gbase + g_tid) + (176 * k1_x1_k0_x0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)4224, (int)12847)];
          }
        }
      }
      {
        int gbase = ((g_gid + (k1_gid * 176)) + (k0_gid * 1232));
        int g_tid = (tid % 32);
        int k1_k0_tid = ((tid / 32) % 8);
        for (int k1_k0_lid = 0; k1_k0_lid < 7; k1_k0_lid += 1)
        {
          int k1_k0_cond = ((k1_k0_lid < 6) || (k1_k0_tid < 1));
          if (k1_k0_cond)
          {
            int k1_k0 = ((8 * k1_k0_lid) + k1_k0_tid);
            int lidx = ((49 * g_tid) + k1_k0);
            int gidx = ((gbase + g_tid) + (176 * k1_k0));
            in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)8623)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if ((((((-1 * k0_gid) <= -3) && ((((k0_gid + 7) - 1) + 6) <= 9)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -3)) && ((((k1_gid + 7) - 1) + ((x1_gid + 4) - 1)) <= 9)))
      {
        for (int k0_lid = 0; k0_lid < 7; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 7; k1_lid += 1)
          {
            int g_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
            {
              int x0_cond = ((x0_lid < 3) || (x0_tid < 1));
              int x0 = select((int)0, (int)((2 * x0_lid) + x0_tid), (int)x0_cond);
              float val1 = in1_shared[(((((95 * g_tid) + k1_lid) + x1_tid) + (7 * k0_lid)) + (7 * x0))];
              float val2 = in2_shared[(((49 * g_tid) + k1_lid) + (7 * k0_lid))];
              float agg_rhs = mad(val2, val1, agg[x0_lid]);
              agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)x0_cond);
            }
          }
        }
      }
      else
      {
        for (int k0_lid = 0; k0_lid < 7; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 7; k1_lid += 1)
          {
            int g_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
            {
              int x0_cond = ((x0_lid < 3) || (x0_tid < 1));
              int x0 = select((int)0, (int)((2 * x0_lid) + x0_tid), (int)x0_cond);
              float val1 = in1_shared[(((((95 * g_tid) + k1_lid) + x1_tid) + (7 * k0_lid)) + (7 * x0))];
              float val2 = in2_shared[(((49 * g_tid) + k1_lid) + (7 * k0_lid))];
              float agg_rhs = mad(val2, val1, agg[x0_lid]);
              agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)(x0_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * x0)) <= -3) && (((k0_gid + k0_lid) + x0) <= 9)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1_tid))) <= -3)) && (((k1_gid + k1_lid) + (x1_gid + x1_tid)) <= 9))));
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int g_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 4);
  int x0_tid = ((tid / 128) % 2);
  int g_cond = ((g_gid != 160) || (g_tid < 16));
  if (g_cond)
  {
    int x1_cond = ((x1_gid != 4) || (x1_tid < 3));
    if (x1_cond)
    {
      for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 3) || (x0_tid < 1));
        if (x0_cond)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float LX_T2174 = agg[x0_lid];
          int gout_idx = (((g_gid + g_tid) + (1232 * x0)) + (176 * (x1_gid + x1_tid)));
          if (((gout_idx >= 0) && (gout_idx < 8624)))
          {
            X_T2174[gout_idx] = LX_T2174;
          }
        }
      }
    }
  }
}
