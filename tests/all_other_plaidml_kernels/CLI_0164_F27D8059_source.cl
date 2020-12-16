#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2816 11 1
// lid: 256 1 1
// Original:
// X_T647[n, x0, x1, c : _T983, _T984, _T985, _T986] = +(X_T646[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T647[n, x0, x1, c : _T983, _T984, _T985, _T986] = +(X_T646[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 42, 0 <= x1 < 42, 0 <= -1 + k0 + x0 < 42, 0 <= -1 + k1 + x1 < 42, 0 <= c < 168, 0 <= c < 168, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 42, 0 <= x1 < 42, 0 <= -1 + k0 + x0 < 42, 0 <= -1 + k1 + x1 < 42, 0 <= c < 168 }
// Defracted:
// X_T647[n, x0, x1, c : _T983, _T984, _T985, _T986] = +(X_T646[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T647    X_T646  
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
// Elementwise input X_T645 shape: fp32(1, 42, 42, 168):(296352, 7056, 168, 1):1157.62 KiB
// Elementwise input X_T508 shape: fp32(1, 42, 42, 168):(296352, 7056, 168, 1):1157.62 KiB
// Elementwise op: [[pid(normal_A_block_0)]] X_T648 = div(X_T645, X_T647)
// Elementwise op: [[pid(Add)]] X_T649 = add(X_T648, X_T508)
// Tile size: { 168, 3, 3, 4, 4 }
// Contraction output var shape: fp32(1, 42, 42, 168):(296352, 7056, 168, 1):1157.62 KiB
// Computed true ops: 10668672
// Computed work groups: 121
// Computed inner loops: 1
// Computed shared mem: 24216
// Computed out regs: 12288
// Computed mem read: 24960
// Computed mem write: 24576
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2816, 11, 1
__kernel void kernel_c42_sdk_231(__global float* restrict  X_T647, __global float* restrict  X_T649, __global const float* restrict  in1, __global const float* restrict  X_T645, __global const float* restrict  X_T508)
{
  in1 = (in1 + -7224);
  int tid = get_local_id(0);
  float agg[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[6054];
  int x1_gid = (get_group_id(0) * 4);
  int x0_gid = (get_group_id(1) * 4);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((((k1_gid * 168) + (x1_gid * 168)) + (k0_gid * 7056)) + (x0_gid * 7056));
        int c_k1_x1_tid = (tid % 256);
        for (int c_k1_x1_lid = 0; c_k1_x1_lid < 4; c_k1_x1_lid += 1)
        {
          int c_k1_x1_cond = ((c_k1_x1_lid < 3) || (c_k1_x1_tid < 240));
          if (c_k1_x1_cond)
          {
            int c_k1_x1 = ((256 * c_k1_x1_lid) + c_k1_x1_tid);
            for (int k0_x0_lid = 0; k0_x0_lid < 6; k0_x0_lid += 1)
            {
              int lidx = (c_k1_x1 + (1009 * k0_x0_lid));
              int gidx = ((gbase + c_k1_x1) + (7056 * k0_x0_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)7224, (int)303575)];
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -1) && ((((k0_gid + 3) - 1) + ((x0_gid + 4) - 1)) <= 42)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -1)) && ((((k1_gid + 3) - 1) + ((x1_gid + 4) - 1)) <= 42)))
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int c_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
            {
              int x0 = ((2 * x0_lid) + x0_tid);
              for (int c_lid = 0; c_lid < 6; c_lid += 1)
              {
                int c_cond = ((c_lid < 5) || (c_tid < 8));
                int c = select((int)0, (int)((32 * c_lid) + c_tid), (int)c_cond);
                float val1 = in1_shared[((((c + (168 * k1_lid)) + (168 * x1_tid)) + (1009 * k0_lid)) + (1009 * x0))];
                int agg_idx = (c_lid + (x0_lid * 6));
                float agg_rhs = (agg[agg_idx] + val1);
                agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)c_cond);
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
            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
            {
              int x0 = ((2 * x0_lid) + x0_tid);
              for (int c_lid = 0; c_lid < 6; c_lid += 1)
              {
                int c_cond = ((c_lid < 5) || (c_tid < 8));
                int c = select((int)0, (int)((32 * c_lid) + c_tid), (int)c_cond);
                float val1 = in1_shared[((((c + (168 * k1_lid)) + (168 * x1_tid)) + (1009 * k0_lid)) + (1009 * x0))];
                int agg_idx = (c_lid + (x0_lid * 6));
                float agg_rhs = (agg[agg_idx] + val1);
                agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)(c_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * (x0_gid + x0))) <= -1) && (((k0_gid + k0_lid) + (x0_gid + x0)) <= 42)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1_tid))) <= -1)) && (((k1_gid + k1_lid) + (x1_gid + x1_tid)) <= 42))));
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
  int x1_cond = ((x1_gid != 40) || (x1_tid < 2));
  if (x1_cond)
  {
    for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
    {
      int x0_cond = ((x0_lid < 1) || (x0_gid != 40));
      if (x0_cond)
      {
        int x0 = ((2 * x0_lid) + x0_tid);
        for (int c_lid = 0; c_lid < 6; c_lid += 1)
        {
          int c_cond = ((c_lid < 5) || (c_tid < 8));
          if (c_cond)
          {
            int c = ((32 * c_lid) + c_tid);
            float LX_T647 = agg[(c_lid + (x0_lid * 6))];
            int gout_idx = ((c + (7056 * (x0_gid + x0))) + (168 * (x1_gid + x1_tid)));
            if (((gout_idx >= 0) && (gout_idx < 296352)))
            {
              float LX_T645 = X_T645[gout_idx];
              float LX_T508 = X_T508[gout_idx];
              float LX_T648 = (LX_T645 / LX_T647);
              float LX_T649 = (LX_T648 + LX_T508);
              X_T647[gout_idx] = LX_T647;
              X_T649[gout_idx] = LX_T649;
            }
          }
        }
      }
    }
  }
}
