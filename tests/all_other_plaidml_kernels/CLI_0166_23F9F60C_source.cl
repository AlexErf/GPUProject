#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 3 11
// lid: 256 1 1
// Original:
// X_T654[n, x0, x1, c : _T1006, _T1007, _T1008, _T1009] = +(X_T653[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T654[n, x0, x1, c : _T1006, _T1007, _T1008, _T1009] = +(X_T653[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 42, 0 <= x1 < 42, 0 <= -1 + k0 + x0 < 42, 0 <= -1 + k1 + x1 < 42, 0 <= c < 168, 0 <= c < 168, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 42, 0 <= x1 < 42, 0 <= -1 + k0 + x0 < 42, 0 <= -1 + k1 + x1 < 42, 0 <= c < 168 }
// Defracted:
// X_T654[n, x0, x1, c : _T1006, _T1007, _T1008, _T1009] = +(X_T653[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T654    X_T653  
//        c       168         1         1  
//       k0         3         0      7056  
//       k1         3         0       168  
//       x0        42      7056      7056  
//       x1        42       168       168  
//      off                   0     -7224  
//      vec                   1         1  
// Constraint: (0,-1,0,-1,0) <= -1
// Constraint: (0,1,0,1,0) <= 42
// Constraint: (0,0,-1,0,-1) <= -1
// Constraint: (0,0,1,0,1) <= 42
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 168, 3, 3, 42, 42 }
// Out stride: { 1, 0, 0, 7056, 168 }
// Input 1 offset: -7224
// Input 1 stride: { 1, 7056, 168, 7056, 168 }
// Elementwise input X_T652 shape: fp32(1, 42, 42, 168):(296352, 7056, 168, 1):1157.62 KiB
// Elementwise op: [[pid(normal_A_block_0)]] X_T655 = div(X_T652, X_T654)
// Elementwise op: [[pid(Add)]] X_T656 = add(X_T655, X_T655)
// Tile size: { 64, 3, 3, 4, 16 }
// Contraction output var shape: fp32(1, 42, 42, 168):(296352, 7056, 168, 1):1157.62 KiB
// Computed true ops: 10668672
// Computed work groups: 99
// Computed inner loops: 1
// Computed shared mem: 28104
// Computed out regs: 16384
// Computed mem read: 28160
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 3, 11
__kernel void kernel_c42_sdk_236(__global float* restrict  X_T656, __global const float* restrict  in1, __global const float* restrict  X_T652)
{
  in1 = (in1 + -7224);
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[7026];
  int c_gid = (get_group_id(0) * 64);
  int x1_gid = (get_group_id(1) * 16);
  int x0_gid = (get_group_id(2) * 4);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((((c_gid + (k1_gid * 168)) + (x1_gid * 168)) + (k0_gid * 7056)) + (x0_gid * 7056));
        int c_tid = (tid % 64);
        int k0_x0_tid = ((tid / 64) % 4);
        for (int k0_x0_lid = 0; k0_x0_lid < 2; k0_x0_lid += 1)
        {
          int k0_x0_cond = ((k0_x0_lid < 1) || (k0_x0_tid < 2));
          if (k0_x0_cond)
          {
            int k0_x0 = ((4 * k0_x0_lid) + k0_x0_tid);
            for (int k1_x1_lid = 0; k1_x1_lid < 18; k1_x1_lid += 1)
            {
              int lidx = ((c_tid + (1171 * k0_x0)) + (65 * k1_x1_lid));
              int gidx = (((gbase + c_tid) + (7056 * k0_x0)) + (168 * k1_x1_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)7224, (int)303575)];
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -1) && ((((k0_gid + 3) - 1) + ((x0_gid + 4) - 1)) <= 42)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -1)) && ((((k1_gid + 3) - 1) + ((x1_gid + 16) - 1)) <= 42)))
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int c_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int c_lid = 0; c_lid < 2; c_lid += 1)
            {
              int c = ((32 * c_lid) + c_tid);
              for (int x1_lid = 0; x1_lid < 4; x1_lid += 1)
              {
                int x1 = ((4 * x1_lid) + x1_tid);
                for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
                {
                  int x0 = ((2 * x0_lid) + x0_tid);
                  float val1 = in1_shared[((((c + (65 * k1_lid)) + (65 * x1)) + (1171 * k0_lid)) + (1171 * x0))];
                  int agg_idx = ((c_lid + (x1_lid * 2)) + (x0_lid * 8));
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
            for (int c_lid = 0; c_lid < 2; c_lid += 1)
            {
              int c = ((32 * c_lid) + c_tid);
              for (int x1_lid = 0; x1_lid < 4; x1_lid += 1)
              {
                int x1 = ((4 * x1_lid) + x1_tid);
                for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
                {
                  int x0 = ((2 * x0_lid) + x0_tid);
                  float val1 = in1_shared[((((c + (65 * k1_lid)) + (65 * x1)) + (1171 * k0_lid)) + (1171 * x0))];
                  int agg_idx = ((c_lid + (x1_lid * 2)) + (x0_lid * 8));
                  float agg_rhs = (agg[agg_idx] + val1);
                  agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)((((((-1 * (k0_gid + k0_lid)) + (-1 * (x0_gid + x0))) <= -1) && (((k0_gid + k0_lid) + (x0_gid + x0)) <= 42)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1))) <= -1)) && (((k1_gid + k1_lid) + (x1_gid + x1)) <= 42)));
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
  for (int c_lid = 0; c_lid < 2; c_lid += 1)
  {
    int c_cond = ((c_lid < 1) || ((c_gid != 128) || (c_tid < 8)));
    if (c_cond)
    {
      int c = ((32 * c_lid) + c_tid);
      for (int x1_lid = 0; x1_lid < 4; x1_lid += 1)
      {
        int x1_cond = (((x1_lid < 2) || ((x1_gid != 32) || (x1_tid < 2))) && ((x1_lid < 3) || (x1_gid != 32)));
        if (x1_cond)
        {
          int x1 = ((4 * x1_lid) + x1_tid);
          for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
          {
            int x0_cond = ((x0_lid < 1) || (x0_gid != 40));
            if (x0_cond)
            {
              int x0 = ((2 * x0_lid) + x0_tid);
              float LX_T654 = agg[((c_lid + (x1_lid * 2)) + (x0_lid * 8))];
              int gout_idx = (((c_gid + c) + (7056 * (x0_gid + x0))) + (168 * (x1_gid + x1)));
              if (((gout_idx >= 0) && (gout_idx < 296352)))
              {
                float LX_T652 = X_T652[gout_idx];
                float LX_T655 = (LX_T652 / LX_T654);
                float LX_T656 = (LX_T655 + LX_T655);
                X_T656[gout_idx] = LX_T656;
              }
            }
          }
        }
      }
    }
  }
}
