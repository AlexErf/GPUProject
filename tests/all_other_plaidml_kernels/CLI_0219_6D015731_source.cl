#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 11 1
// lid: 256 1 1
// Original:
// X_T1922[n, x0, x1, c : _T3031, _T3032, _T3033, _T3034] = +(X_T1801[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T1922[n, x0, x1, c : _T3031, _T3032, _T3033, _T3034] = +(X_T1801[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 21, 0 <= x1 < 21, 0 <= -1 + k0 + x0 < 21, 0 <= -1 + k1 + x1 < 21, 0 <= c < 336, 0 <= c < 336, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 21, 0 <= x1 < 21, 0 <= -1 + k0 + x0 < 21, 0 <= -1 + k1 + x1 < 21, 0 <= c < 336 }
// Defracted:
// X_T1922[n, x0, x1, c : _T3031, _T3032, _T3033, _T3034] = +(X_T1801[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T1922   X_T1801  
//        c       336         1         1  
//       k0         3         0      7056  
//       k1         3         0       336  
//       x0        21      7056      7056  
//       x1        21       336       336  
//      off                   0     -7392  
//      vec                   1         1  
// Constraint: (0,-1,0,-1,0) <= -1
// Constraint: (0,1,0,1,0) <= 21
// Constraint: (0,0,-1,0,-1) <= -1
// Constraint: (0,0,1,0,1) <= 21
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 336, 3, 3, 21, 21 }
// Out stride: { 1, 0, 0, 7056, 336 }
// Input 1 offset: -7392
// Input 1 stride: { 1, 7056, 336, 7056, 336 }
// Elementwise input X_T1707 shape: fp32(1, 21, 21, 336):(148176, 7056, 336, 1):578.812 KiB
// Elementwise input X_T1789 shape: fp32(1, 21, 21, 336):(148176, 7056, 336, 1):578.812 KiB
// Elementwise op: [[pid(normal_A_block_7)]] X_T1923 = div(X_T1922, X_T1707)
// Elementwise op: [[pid(Add)]] X_T1924 = add(X_T1923, X_T1789)
// Tile size: { 32, 3, 3, 21, 4 }
// Contraction output var shape: fp32(1, 21, 21, 336):(148176, 7056, 336, 1):578.812 KiB
// Computed true ops: 5334336
// Computed work groups: 66
// Computed inner loops: 1
// Computed shared mem: 17688
// Computed out regs: 11264
// Computed mem read: 18336
// Computed mem write: 10752
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 11, 1
__kernel void kernel_c42_sdk_733(__global float* restrict  X_T1924, __global const float* restrict  in1, __global const float* restrict  X_T1707, __global const float* restrict  X_T1789)
{
  in1 = (in1 + -7392);
  int tid = get_local_id(0);
  float agg[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[4422];
  int c_gid = (get_group_id(1) * 32);
  int x1_gid = (get_group_id(0) * 4);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = (((c_gid + (k1_gid * 336)) + (x1_gid * 336)) + (k0_gid * 7056));
        int c_tid = (tid % 32);
        int k1_x1_tid = ((tid / 32) % 8);
        int k1_x1_cond = (k1_x1_tid < 6);
        if (k1_x1_cond)
        {
          for (int k0_x0_lid = 0; k0_x0_lid < 23; k0_x0_lid += 1)
          {
            int lidx = (((23 * c_tid) + (737 * k1_x1_tid)) + k0_x0_lid);
            int gidx = (((gbase + c_tid) + (336 * k1_x1_tid)) + (7056 * k0_x0_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)7392, (int)155567)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if ((((((-1 * k0_gid) <= -1) && ((((k0_gid + 3) - 1) + 20) <= 21)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -1)) && ((((k1_gid + 3) - 1) + ((x1_gid + 4) - 1)) <= 21)))
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int c_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int x0_lid = 0; x0_lid < 11; x0_lid += 1)
            {
              int x0_cond = ((x0_lid < 10) || (x0_tid < 1));
              int x0 = select((int)0, (int)((2 * x0_lid) + x0_tid), (int)x0_cond);
              float val1 = in1_shared[(((((23 * c_tid) + (737 * k1_lid)) + (737 * x1_tid)) + k0_lid) + x0)];
              float agg_rhs = (agg[x0_lid] + val1);
              agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)x0_cond);
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
            for (int x0_lid = 0; x0_lid < 11; x0_lid += 1)
            {
              int x0_cond = ((x0_lid < 10) || (x0_tid < 1));
              int x0 = select((int)0, (int)((2 * x0_lid) + x0_tid), (int)x0_cond);
              float val1 = in1_shared[(((((23 * c_tid) + (737 * k1_lid)) + (737 * x1_tid)) + k0_lid) + x0)];
              float agg_rhs = (agg[x0_lid] + val1);
              agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)(x0_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * x0)) <= -1) && (((k0_gid + k0_lid) + x0) <= 21)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1_tid))) <= -1)) && (((k1_gid + k1_lid) + (x1_gid + x1_tid)) <= 21))));
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
  int c_cond = ((c_gid != 320) || (c_tid < 16));
  if (c_cond)
  {
    int x1_cond = ((x1_gid != 20) || (x1_tid < 1));
    if (x1_cond)
    {
      for (int x0_lid = 0; x0_lid < 11; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 10) || (x0_tid < 1));
        if (x0_cond)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float LX_T1922 = agg[x0_lid];
          int gout_idx = (((c_gid + c_tid) + (7056 * x0)) + (336 * (x1_gid + x1_tid)));
          if (((gout_idx >= 0) && (gout_idx < 148176)))
          {
            float LX_T1707 = X_T1707[gout_idx];
            float LX_T1789 = X_T1789[gout_idx];
            float LX_T1923 = (LX_T1922 / LX_T1707);
            float LX_T1924 = (LX_T1923 + LX_T1789);
            X_T1924[gout_idx] = LX_T1924;
          }
        }
      }
    }
  }
}
