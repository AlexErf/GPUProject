#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 4096 1 1
// lid: 256 1 1
// Original:
// X_T419[n, x0, x1, g, gco : _T476, _T477, _T478, _T479, _T480] = +(X_T418[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T25[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T419[n, x0, x1, g, gco : _T476, _T477, _T478, _T479, _T480] = +(X_T418[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T25[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 7, 0 <= x1 < 7, 0 <= -1 + k0 + x0 < 7, 0 <= -1 + k1 + x1 < 7, 0 <= g < 1024, 0 <= g + gci < 1024, 0 <= g < 1024, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 7, 0 <= x1 < 7, 0 <= -1 + k0 + x0 < 7, 0 <= -1 + k1 + x1 < 7, 0 <= g < 1024, 0 <= g + gci < 1024 }
// Defracted:
// X_T419[n, x0, x1, g, gco : _T476, _T477, _T478, _T479, _T480] = +(X_T418[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T25[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T419    X_T418     X_T25  
//        g      1024         1         1         1  
//       k0         3         0      7168      3072  
//       k1         3         0      1024      1024  
//       x0         7      7168      7168         0  
//       x1         7      1024      1024         0  
//      off                   0     -8192         0  
//      vec                   1         1         1  
// Constraint: (0,-1,0,-1,0) <= -1
// Constraint: (0,1,0,1,0) <= 7
// Constraint: (0,0,-1,0,-1) <= -1
// Constraint: (0,0,1,0,1) <= 7
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 1024, 3, 3, 7, 7 }
// Out stride: { 1, 0, 0, 7168, 1024 }
// Input 1 offset: -8192
// Input 1 stride: { 1, 7168, 1024, 7168, 1024 }
// Input 2 offset: 0
// Input 2 stride: { 1, 3072, 1024, 0, 0 }
// Tile size: { 64, 3, 3, 7, 7 }
// Contraction output var shape: fp32(1, 7, 7, 1024, 1):(50176, 7168, 1024, 1, 1):196 KiB
// Computed true ops: 903168
// Computed work groups: 16
// Computed inner loops: 1
// Computed shared mem: 18944
// Computed out regs: 16384
// Computed mem read: 18944
// Computed mem write: 12544
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 4096, 1, 1
__kernel void kernel_c25_sdk_106(__global float* restrict  X_T419, __global const float* restrict  in1, __global const float* restrict  in2)
{
  in1 = (in1 + -8192);
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[4160];
  __local float in2_shared[576];
  int g_gid = (get_group_id(0) * 64);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((g_gid + (k1_gid * 1024)) + (k0_gid * 7168));
        int g_tid = (tid % 64);
        int k1_x1_k0_x0_tid = ((tid / 64) % 4);
        for (int k1_x1_k0_x0_lid = 0; k1_x1_k0_x0_lid < 17; k1_x1_k0_x0_lid += 1)
        {
          int k1_x1_k0_x0_cond = ((k1_x1_k0_x0_lid < 16) || (k1_x1_k0_x0_tid < 1));
          if (k1_x1_k0_x0_cond)
          {
            int k1_x1_k0_x0 = ((4 * k1_x1_k0_x0_lid) + k1_x1_k0_x0_tid);
            int lidx = ((65 * g_tid) + k1_x1_k0_x0);
            int gidx = ((gbase + g_tid) + (1024 * k1_x1_k0_x0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)8192, (int)58367)];
          }
        }
      }
      {
        int gbase = ((g_gid + (k1_gid * 1024)) + (k0_gid * 3072));
        int g_tid = (tid % 64);
        int k1_k0_tid = ((tid / 64) % 4);
        for (int k1_k0_lid = 0; k1_k0_lid < 3; k1_k0_lid += 1)
        {
          int k1_k0_cond = ((k1_k0_lid < 2) || (k1_k0_tid < 1));
          if (k1_k0_cond)
          {
            int k1_k0 = ((4 * k1_k0_lid) + k1_k0_tid);
            int lidx = ((9 * g_tid) + k1_k0);
            int gidx = ((gbase + g_tid) + (1024 * k1_k0));
            in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)9215)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if ((((((-1 * k0_gid) <= -1) && ((((k0_gid + 3) - 1) + 6) <= 7)) && ((-1 * k1_gid) <= -1)) && ((((k1_gid + 3) - 1) + 6) <= 7)))
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int g_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
            {
              int x1_cond = ((x1_lid < 1) || (x1_tid < 3));
              int x1 = select((int)0, (int)((4 * x1_lid) + x1_tid), (int)x1_cond);
              for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
              {
                int x0_cond = ((x0_lid < 3) || (x0_tid < 1));
                int x0 = select((int)0, (int)((2 * x0_lid) + x0_tid), (int)(x1_cond && x0_cond));
                for (int g_lid = 0; g_lid < 2; g_lid += 1)
                {
                  int g = ((32 * g_lid) + g_tid);
                  float val1 = in1_shared[(((((65 * g) + k1_lid) + x1) + (7 * k0_lid)) + (7 * x0))];
                  float val2 = in2_shared[(((9 * g) + k1_lid) + (3 * k0_lid))];
                  int agg_idx = ((g_lid + (x1_lid * 2)) + (x0_lid * 4));
                  float agg_rhs = mad(val2, val1, agg[agg_idx]);
                  agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)(x1_cond && x0_cond));
                }
              }
            }
          }
        }
      }
      else
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int g_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
            {
              int x1_cond = ((x1_lid < 1) || (x1_tid < 3));
              int x1 = select((int)0, (int)((4 * x1_lid) + x1_tid), (int)x1_cond);
              for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
              {
                int x0_cond = ((x0_lid < 3) || (x0_tid < 1));
                int x0 = select((int)0, (int)((2 * x0_lid) + x0_tid), (int)(x1_cond && x0_cond));
                for (int g_lid = 0; g_lid < 2; g_lid += 1)
                {
                  int g = ((32 * g_lid) + g_tid);
                  float val1 = in1_shared[(((((65 * g) + k1_lid) + x1) + (7 * k0_lid)) + (7 * x0))];
                  float val2 = in2_shared[(((9 * g) + k1_lid) + (3 * k0_lid))];
                  int agg_idx = ((g_lid + (x1_lid * 2)) + (x0_lid * 4));
                  float agg_rhs = mad(val2, val1, agg[agg_idx]);
                  agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)((x1_cond && x0_cond) && ((((((-1 * (k0_gid + k0_lid)) + (-1 * x0)) <= -1) && (((k0_gid + k0_lid) + x0) <= 7)) && (((-1 * (k1_gid + k1_lid)) + (-1 * x1)) <= -1)) && (((k1_gid + k1_lid) + x1) <= 7))));
                }
              }
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
  for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
  {
    int x1_cond = ((x1_lid < 1) || (x1_tid < 3));
    if (x1_cond)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 3) || (x0_tid < 1));
        if (x0_cond)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          for (int g_lid = 0; g_lid < 2; g_lid += 1)
          {
            int g = ((32 * g_lid) + g_tid);
            float LX_T419 = agg[((g_lid + (x1_lid * 2)) + (x0_lid * 4))];
            int gout_idx = (((g_gid + g) + (7168 * x0)) + (1024 * x1));
            if (((gout_idx >= 0) && (gout_idx < 50176)))
            {
              X_T419[gout_idx] = LX_T419;
            }
          }
        }
      }
    }
  }
}
