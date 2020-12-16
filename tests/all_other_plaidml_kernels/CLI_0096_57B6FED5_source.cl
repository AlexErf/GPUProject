#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Original:
// X_T209[n, x0, x1, g, gco : _T279, _T280, _T281, _T282, _T283] = +(X_T208[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T206[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T209[n, x0, x1, g, gco : _T279, _T280, _T281, _T282, _T283] = +(X_T208[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T206[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= g < 11, 0 <= g + gci < 11, 0 <= g < 11, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= -1 + k0 + x0 < 56, 0 <= -1 + k1 + x1 < 56, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= g < 11, 0 <= g + gci < 11, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= -1 + k0 + x0 < 56, 0 <= -1 + k1 + x1 < 56 }
// Defracted:
// X_T209[n, x0, x1, g, gco : _T279, _T280, _T281, _T282, _T283] = +(X_T208[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T206[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T209    X_T208    X_T206  
//        g        11         1         1         1  
//       k0         3         0       616        33  
//       k1         3         0        11        11  
//       x0        56       616       616         0  
//       x1        56        11        11         0  
//      off                   0      -627         0  
//      vec                   1         1         1  
// Constraint: (0,-1,0,-1,0) <= -1
// Constraint: (0,1,0,1,0) <= 56
// Constraint: (0,0,-1,0,-1) <= -1
// Constraint: (0,0,1,0,1) <= 56
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 11, 3, 3, 56, 56 }
// Out stride: { 1, 0, 0, 616, 11 }
// Input 1 offset: -627
// Input 1 stride: { 1, 616, 11, 616, 11 }
// Input 2 offset: 0
// Input 2 stride: { 1, 33, 11, 0, 0 }
// Tile size: { 11, 3, 3, 8, 8 }
// Contraction output var shape: fp32(1, 56, 56, 11, 1):(34496, 616, 11, 1, 1):134.75 KiB
// Computed true ops: 620928
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 4836
// Computed out regs: 4096
// Computed mem read: 4736
// Computed mem write: 8192
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c42_sdk_64(__global float* restrict  X_T209, __global const float* restrict  in1, __global const float* restrict  in2)
{
  in1 = (in1 + -627);
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[1110];
  __local float in2_shared[99];
  int x1_gid = (get_group_id(0) * 8);
  int x0_gid = (get_group_id(1) * 8);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((((k1_gid * 11) + (x1_gid * 11)) + (k0_gid * 616)) + (x0_gid * 616));
        int g_k1_x1_tid = (tid % 128);
        int k0_x0_tid = ((tid / 128) % 2);
        int g_k1_x1_cond = (g_k1_x1_tid < 110);
        if (g_k1_x1_cond)
        {
          for (int k0_x0_lid = 0; k0_x0_lid < 5; k0_x0_lid += 1)
          {
            int k0_x0 = ((2 * k0_x0_lid) + k0_x0_tid);
            int lidx = (g_k1_x1_tid + (111 * k0_x0));
            int gidx = ((gbase + g_k1_x1_tid) + (616 * k0_x0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)627, (int)35122)];
          }
        }
      }
      {
        int gbase = ((k1_gid * 11) + (k0_gid * 33));
        int g_k1_k0_tid = (tid % 128);
        int g_k1_k0_cond = (g_k1_k0_tid < 99);
        if (g_k1_k0_cond)
        {
          if ((tid < 128))
          {
            int gidx = (gbase + g_k1_k0_tid);
            in2_shared[g_k1_k0_tid] = in2[clamp((int)gidx, (int)0, (int)98)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -1) && ((((k0_gid + 3) - 1) + ((x0_gid + 8) - 1)) <= 56)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -1)) && ((((k1_gid + 3) - 1) + ((x1_gid + 8) - 1)) <= 56)))
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int g_tid = (tid % 16);
            int x1_tid = ((tid / 16) % 4);
            int x0_tid = ((tid / 64) % 4);
            int g_cond = (g_tid < 11);
            int g = select((int)0, (int)g_tid, (int)g_cond);
            for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
            {
              int x1 = ((4 * x1_lid) + x1_tid);
              for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
              {
                int x0 = ((4 * x0_lid) + x0_tid);
                float val1 = in1_shared[((((g + (11 * k1_lid)) + (11 * x1)) + (111 * k0_lid)) + (111 * x0))];
                float val2 = in2_shared[((g + (11 * k1_lid)) + (33 * k0_lid))];
                int agg_idx = (x1_lid + (x0_lid * 2));
                float agg_rhs = mad(val2, val1, agg[agg_idx]);
                agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)g_cond);
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
            int g_tid = (tid % 16);
            int x1_tid = ((tid / 16) % 4);
            int x0_tid = ((tid / 64) % 4);
            int g_cond = (g_tid < 11);
            int g = select((int)0, (int)g_tid, (int)g_cond);
            for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
            {
              int x1 = ((4 * x1_lid) + x1_tid);
              for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
              {
                int x0 = ((4 * x0_lid) + x0_tid);
                float val1 = in1_shared[((((g + (11 * k1_lid)) + (11 * x1)) + (111 * k0_lid)) + (111 * x0))];
                float val2 = in2_shared[((g + (11 * k1_lid)) + (33 * k0_lid))];
                int agg_idx = (x1_lid + (x0_lid * 2));
                float agg_rhs = mad(val2, val1, agg[agg_idx]);
                agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)(g_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * (x0_gid + x0))) <= -1) && (((k0_gid + k0_lid) + (x0_gid + x0)) <= 56)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1))) <= -1)) && (((k1_gid + k1_lid) + (x1_gid + x1)) <= 56))));
              }
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int g_tid = (tid % 16);
  int x1_tid = ((tid / 16) % 4);
  int x0_tid = ((tid / 64) % 4);
  int g_cond = (g_tid < 11);
  if (g_cond)
  {
    for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
      {
        int x0 = ((4 * x0_lid) + x0_tid);
        float LX_T209 = agg[(x1_lid + (x0_lid * 2))];
        int gout_idx = ((g_tid + (616 * (x0_gid + x0))) + (11 * (x1_gid + x1)));
        if (((gout_idx >= 0) && (gout_idx < 34496)))
        {
          X_T209[gout_idx] = LX_T209;
        }
      }
    }
  }
}
