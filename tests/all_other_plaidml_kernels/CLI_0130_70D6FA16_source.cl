#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 5 5
// lid: 256 1 1
// Original:
// X_T162[n, x0, x1, c : _T183, _T184, _T185, _T186] = +(X_T161[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T162[n, x0, x1, c : _T183, _T184, _T185, _T186] = +(X_T161[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 35, 0 <= x1 < 35, 0 <= -1 + k0 + x0 < 35, 0 <= -1 + k1 + x1 < 35, 0 <= c < 192, 0 <= c < 192, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 35, 0 <= x1 < 35, 0 <= -1 + k0 + x0 < 35, 0 <= -1 + k1 + x1 < 35, 0 <= c < 192 }
// Defracted:
// X_T162[n, x0, x1, c : _T183, _T184, _T185, _T186] = +(X_T161[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T162    X_T161  
//        c       192         1         1  
//       k0         3         0      6720  
//       k1         3         0       192  
//       x0        35      6720      6720  
//       x1        35       192       192  
//      off                   0     -6912  
//      vec                   1         1  
// Constraint: (0,-1,0,-1,0) <= -1
// Constraint: (0,1,0,1,0) <= 35
// Constraint: (0,0,-1,0,-1) <= -1
// Constraint: (0,0,1,0,1) <= 35
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 192, 3, 3, 35, 35 }
// Out stride: { 1, 0, 0, 6720, 192 }
// Input 1 offset: -6912
// Input 1 stride: { 1, 6720, 192, 6720, 192 }
// Elementwise input X_T159 shape: fp32(1, 35, 35, 192):(235200, 6720, 192, 1):918.75 KiB
// Elementwise op: [[pid(average_pooling2d_1)]] X_T163 = div(X_T159, X_T162)
// Tile size: { 64, 3, 3, 8, 8 }
// Contraction output var shape: fp32(1, 35, 35, 192):(235200, 6720, 192, 1):918.75 KiB
// Computed true ops: 6350400
// Computed work groups: 75
// Computed inner loops: 1
// Computed shared mem: 26040
// Computed out regs: 16384
// Computed mem read: 26112
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 5, 5
__kernel void kernel_c56_sdk_42(__global float* restrict  X_T163, __global const float* restrict  in1, __global const float* restrict  X_T159)
{
  in1 = (in1 + -6912);
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[6510];
  int c_gid = (get_group_id(0) * 64);
  int x1_gid = (get_group_id(1) * 8);
  int x0_gid = (get_group_id(2) * 8);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((((c_gid + (k1_gid * 192)) + (x1_gid * 192)) + (k0_gid * 6720)) + (x0_gid * 6720));
        int c_tid = (tid % 64);
        int k1_x1_tid = ((tid / 64) % 4);
        for (int k1_x1_lid = 0; k1_x1_lid < 3; k1_x1_lid += 1)
        {
          int k1_x1_cond = ((k1_x1_lid < 2) || (k1_x1_tid < 2));
          if (k1_x1_cond)
          {
            int k1_x1 = ((4 * k1_x1_lid) + k1_x1_tid);
            for (int k0_x0_lid = 0; k0_x0_lid < 10; k0_x0_lid += 1)
            {
              int lidx = ((c_tid + (65 * k1_x1)) + (651 * k0_x0_lid));
              int gidx = (((gbase + c_tid) + (192 * k1_x1)) + (6720 * k0_x0_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)6912, (int)242111)];
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -1) && ((((k0_gid + 3) - 1) + ((x0_gid + 8) - 1)) <= 35)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -1)) && ((((k1_gid + 3) - 1) + ((x1_gid + 8) - 1)) <= 35)))
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int c_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
            {
              int x1 = ((4 * x1_lid) + x1_tid);
              for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
              {
                int x0 = ((2 * x0_lid) + x0_tid);
                for (int c_lid = 0; c_lid < 2; c_lid += 1)
                {
                  int c = ((32 * c_lid) + c_tid);
                  float val1 = in1_shared[((((c + (65 * k1_lid)) + (65 * x1)) + (651 * k0_lid)) + (651 * x0))];
                  int agg_idx = ((c_lid + (x1_lid * 2)) + (x0_lid * 4));
                  float agg_rhs = (agg[agg_idx] + val1);
                  agg[agg_idx] = agg_rhs;
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
            int c_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
            {
              int x1 = ((4 * x1_lid) + x1_tid);
              for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
              {
                int x0 = ((2 * x0_lid) + x0_tid);
                for (int c_lid = 0; c_lid < 2; c_lid += 1)
                {
                  int c = ((32 * c_lid) + c_tid);
                  float val1 = in1_shared[((((c + (65 * k1_lid)) + (65 * x1)) + (651 * k0_lid)) + (651 * x0))];
                  int agg_idx = ((c_lid + (x1_lid * 2)) + (x0_lid * 4));
                  float agg_rhs = (agg[agg_idx] + val1);
                  agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)((((((-1 * (k0_gid + k0_lid)) + (-1 * (x0_gid + x0))) <= -1) && (((k0_gid + k0_lid) + (x0_gid + x0)) <= 35)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1))) <= -1)) && (((k1_gid + k1_lid) + (x1_gid + x1)) <= 35)));
                }
              }
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 4);
  int x0_tid = ((tid / 128) % 2);
  for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
  {
    int x1_cond = (((x1_lid < 0) || ((x1_gid != 32) || (x1_tid < 3))) && ((x1_lid < 1) || (x1_gid != 32)));
    if (x1_cond)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
      {
        int x0_cond = (((x0_lid < 1) || ((x0_gid != 32) || (x0_tid < 1))) && ((x0_lid < 2) || (x0_gid != 32)));
        if (x0_cond)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          for (int c_lid = 0; c_lid < 2; c_lid += 1)
          {
            int c = ((32 * c_lid) + c_tid);
            float LX_T162 = agg[((c_lid + (x1_lid * 2)) + (x0_lid * 4))];
            int gout_idx = (((c_gid + c) + (6720 * (x0_gid + x0))) + (192 * (x1_gid + x1)));
            if (((gout_idx >= 0) && (gout_idx < 235200)))
            {
              float LX_T159 = X_T159[gout_idx];
              float LX_T163 = (LX_T159 / LX_T162);
              X_T163[gout_idx] = LX_T163;
            }
          }
        }
      }
    }
  }
}
