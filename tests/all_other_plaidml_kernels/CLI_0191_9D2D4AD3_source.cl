#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 28 1
// lid: 256 1 1
// Original:
// X_T84[n, x0, x1, c : _T25, _T26, _T27, _T28] = >(X_T81[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T84[n, x0, x1, c : _T25, _T26, _T27, _T28] = >(X_T81[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= c < 64, 0 <= c < 64, 0 <= k0 + 2*x0 < 114, 0 <= k1 + 2*x1 < 114, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= c < 64, 0 <= k0 + 2*x0 < 114, 0 <= k1 + 2*x1 < 114 }
// Defracted:
// X_T84[n, x0, x1, c : _T25, _T26, _T27, _T28] = >(X_T81[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T84     X_T81  
//        c        64         1         1  
//       k0         3         0      7296  
//       k1         3         0        64  
//       x0        56      3584     14592  
//       x1        56        64       128  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 64, 3, 3, 56, 56 }
// Out stride: { 1, 0, 0, 3584, 64 }
// Input 1 offset: 0
// Input 1 stride: { 1, 7296, 64, 14592, 128 }
// Elementwise input X_I_30 shape: fp32(64):(1):256 bytes
// Elementwise input X_I_29 shape: fp32(64):(1):256 bytes
// Elementwise op: [[pid(Sub)]] X_T91 = sub(X_T84, X_I_30)
// Elementwise op: [[pid(Mul)]] X_T92 = mul(X_T91, X_I_29)
// Tile size: { 64, 3, 3, 8, 2 }
// Contraction output var shape: fp32(1, 56, 56, 64):(200704, 3584, 64, 1):784 KiB
// Computed true ops: 7225344
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 21760
// Computed out regs: 4096
// Computed mem read: 22016
// Computed mem write: 8192
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 28, 1
__kernel void kernel_c124_sdk_5(__global float* restrict  X_T84, __global float* restrict  X_T92, __global const float* restrict  in1, __global const float* restrict  X_I_30, __global const float* restrict  X_I_29)
{
  int tid = get_local_id(0);
  float agg[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, };
  __local float in1_shared[5440];
  int x1_gid = (get_group_id(1) * 2);
  int x0_gid = (get_group_id(0) * 8);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((((k1_gid * 64) + (x1_gid * 128)) + (k0_gid * 7296)) + (x0_gid * 14592));
        int c_k1_x1_tid = (tid % 256);
        for (int c_k1_x1_lid = 0; c_k1_x1_lid < 2; c_k1_x1_lid += 1)
        {
          int c_k1_x1_cond = ((c_k1_x1_lid < 1) || (c_k1_x1_tid < 64));
          if (c_k1_x1_cond)
          {
            int c_k1_x1 = ((256 * c_k1_x1_lid) + c_k1_x1_tid);
            for (int k0_x0_lid = 0; k0_x0_lid < 17; k0_x0_lid += 1)
            {
              int lidx = ((17 * c_k1_x1) + k0_x0_lid);
              int gidx = ((gbase + c_k1_x1) + (7296 * k0_x0_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)831743)];
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
          int x1_tid = ((tid / 32) % 2);
          int x0_tid = ((tid / 64) % 4);
          for (int c_lid = 0; c_lid < 2; c_lid += 1)
          {
            int c = ((32 * c_lid) + c_tid);
            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
            {
              int x0 = ((4 * x0_lid) + x0_tid);
              float val1 = in1_shared[(((((17 * c) + (1088 * k1_lid)) + (2176 * x1_tid)) + k0_lid) + (2 * x0))];
              int agg_idx = (c_lid + (x0_lid * 2));
              float agg_rhs = select((float)agg[agg_idx], (float)val1, (int)(val1 > agg[agg_idx]));
              agg[agg_idx] = agg_rhs;
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 2);
  int x0_tid = ((tid / 64) % 4);
  for (int c_lid = 0; c_lid < 2; c_lid += 1)
  {
    int c = ((32 * c_lid) + c_tid);
    for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
    {
      int x0 = ((4 * x0_lid) + x0_tid);
      float LX_T84 = agg[(c_lid + (x0_lid * 2))];
      LX_T84 = select((float)LX_T84, (float)0, (int)(LX_T84 == (float)-FLT_MAX));
      int gout_idx = ((c + (3584 * (x0_gid + x0))) + (64 * (x1_gid + x1_tid)));
      float LX_I_30 = X_I_30[c];
      float LX_I_29 = X_I_29[c];
      float LX_T91 = (LX_T84 - LX_I_30);
      float LX_T92 = (LX_T91 * LX_I_29);
      X_T84[gout_idx] = LX_T84;
      X_T92[gout_idx] = LX_T92;
    }
  }
}
