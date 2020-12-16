#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 7 1
// lid: 256 1 1
// Original:
// X_T2288[n, x0, x1, c : _T3625, _T3626, _T3627, _T3628] = +(X_T2287[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T2288[n, x0, x1, c : _T3625, _T3626, _T3627, _T3628] = +(X_T2287[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 7, 0 <= x1 < 7, 0 <= -1 + k0 + x0 < 7, 0 <= -1 + k1 + x1 < 7, 0 <= c < 176, 0 <= c < 176, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 7, 0 <= x1 < 7, 0 <= -1 + k0 + x0 < 7, 0 <= -1 + k1 + x1 < 7, 0 <= c < 176 }
// Defracted:
// X_T2288[n, x0, x1, c : _T3625, _T3626, _T3627, _T3628] = +(X_T2287[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T2288   X_T2287  
//        c       176         1         1  
//       k0         3         0      1232  
//       k1         3         0       176  
//       x0         7      1232      1232  
//       x1         7       176       176  
//      off                   0     -1408  
//      vec                   1         1  
// Constraint: (0,-1,0,-1,0) <= -1
// Constraint: (0,1,0,1,0) <= 7
// Constraint: (0,0,-1,0,-1) <= -1
// Constraint: (0,0,1,0,1) <= 7
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 176, 3, 3, 7, 7 }
// Out stride: { 1, 0, 0, 1232, 176 }
// Input 1 offset: -1408
// Input 1 stride: { 1, 1232, 176, 1232, 176 }
// Elementwise input X_T2286 shape: fp32(1, 7, 7, 176):(8624, 1232, 176, 1):33.6875 KiB
// Elementwise input X_T2185 shape: fp32(1, 7, 7, 176):(8624, 1232, 176, 1):33.6875 KiB
// Elementwise op: [[pid(reduction_A_block_reduce_8)]] X_T2289 = div(X_T2286, X_T2288)
// Elementwise op: [[pid(Add)]] X_T2290 = add(X_T2185, X_T2289)
// Tile size: { 64, 3, 3, 7, 1 }
// Contraction output var shape: fp32(1, 7, 7, 176):(8624, 1232, 176, 1):33.6875 KiB
// Computed true ops: 310464
// Computed work groups: 21
// Computed inner loops: 1
// Computed shared mem: 6912
// Computed out regs: 2048
// Computed mem read: 7024
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 7, 1
__kernel void kernel_c42_sdk_875(__global float* restrict  X_T2288, __global float* restrict  X_T2290, __global const float* restrict  in1, __global const float* restrict  X_T2286, __global const float* restrict  X_T2185)
{
  in1 = (in1 + -1408);
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[1728];
  int c_gid = (get_group_id(0) * 64);
  int x1_gid = get_group_id(1);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = (((c_gid + (k1_gid * 176)) + (x1_gid * 176)) + (k0_gid * 1232));
        int c_tid = (tid % 64);
        int k1_x1_tid = ((tid / 64) % 4);
        int k1_x1_cond = (k1_x1_tid < 3);
        if (k1_x1_cond)
        {
          for (int k0_x0_lid = 0; k0_x0_lid < 9; k0_x0_lid += 1)
          {
            int lidx = (((27 * c_tid) + (9 * k1_x1_tid)) + k0_x0_lid);
            int gidx = (((gbase + c_tid) + (176 * k1_x1_tid)) + (1232 * k0_x0_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)1408, (int)10031)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if ((((((-1 * k0_gid) <= -1) && ((((k0_gid + 3) - 1) + 6) <= 7)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -1)) && ((((k1_gid + 3) - 1) + ((x1_gid + 1) - 1)) <= 7)))
      {
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
              float val1 = in1_shared[((((27 * c) + (9 * k1_lid)) + k0_lid) + x0)];
              float agg_rhs = (agg[c_lid] + val1);
              agg[c_lid] = select((float)agg[c_lid], (float)agg_rhs, (int)x0_cond);
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
            int x0_tid = ((tid / 32) % 8);
            for (int c_lid = 0; c_lid < 2; c_lid += 1)
            {
              int c = ((32 * c_lid) + c_tid);
              int x0_cond = (x0_tid < 7);
              int x0 = select((int)0, (int)x0_tid, (int)x0_cond);
              float val1 = in1_shared[((((27 * c) + (9 * k1_lid)) + k0_lid) + x0)];
              float agg_rhs = (agg[c_lid] + val1);
              agg[c_lid] = select((float)agg[c_lid], (float)agg_rhs, (int)(x0_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * x0)) <= -1) && (((k0_gid + k0_lid) + x0) <= 7)) && (((-1 * (k1_gid + k1_lid)) + (-1 * x1_gid)) <= -1)) && (((k1_gid + k1_lid) + x1_gid) <= 7))));
            }
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
        float LX_T2288 = agg[c_lid];
        int gout_idx = (((c_gid + c) + (1232 * x0_tid)) + (176 * x1_gid));
        if (((gout_idx >= 0) && (gout_idx < 8624)))
        {
          float LX_T2286 = X_T2286[gout_idx];
          float LX_T2185 = X_T2185[gout_idx];
          float LX_T2289 = (LX_T2286 / LX_T2288);
          float LX_T2290 = (LX_T2185 + LX_T2289);
          X_T2288[gout_idx] = LX_T2288;
          X_T2290[gout_idx] = LX_T2290;
        }
      }
    }
  }
}
