#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 4096 1 1
// lid: 256 1 1
// Original:
// X_T1167[n, x0, x1, c : _T1672, _T1673, _T1674, _T1675] = +(X_T1166[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2
// With Index Variables Made Integral:
// X_T1167[n, x0, x1, c : _T1672, _T1673, _T1674, _T1675] = +(X_T1166[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 2, 0 <= k1 < 2, 0 <= x0 < 7, 0 <= x1 < 7, 0 <= k0 + 2*x0 < 14, 0 <= k1 + 2*x1 < 14, 0 <= c < 512, 0 <= c < 512, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 2, 0 <= k1 < 2, 0 <= x0 < 7, 0 <= x1 < 7, 0 <= k0 + 2*x0 < 14, 0 <= k1 + 2*x1 < 14, 0 <= c < 512 }
// Defracted:
// X_T1167[n, x0, x1, c : _T1672, _T1673, _T1674, _T1675] = +(X_T1166[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T1167   X_T1166  
//        c       512         1         1  
//       k0         2         0      7168  
//       k1         2         0       512  
//       x0         7      3584     14336  
//       x1         7       512      1024  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 512, 2, 2, 7, 7 }
// Out stride: { 1, 0, 0, 3584, 512 }
// Input 1 offset: 0
// Input 1 stride: { 1, 7168, 512, 14336, 1024 }
// Elementwise input X_T1165 shape: fp32(1, 7, 7, 512):(25088, 3584, 512, 1):98 KiB
// Elementwise input X_I_453 shape: fp32(512):(1):2 KiB
// Elementwise input X_I_452 shape: fp32(512):(1):2 KiB
// Elementwise op: [[pid(pool4_pool)]] X_T1168 = div(X_T1165, X_T1167)
// Elementwise op: [[pid(Sub)]] X_T1174 = sub(X_T1168, X_I_453)
// Elementwise op: [[pid(Mul)]] X_T1175 = mul(X_T1174, X_I_452)
// Tile size: { 32, 2, 2, 7, 7 }
// Contraction output var shape: fp32(1, 7, 7, 512):(25088, 3584, 512, 1):98 KiB
// Computed true ops: 501760
// Computed work groups: 16
// Computed inner loops: 1
// Computed shared mem: 25216
// Computed out regs: 8192
// Computed mem read: 25676
// Computed mem write: 12544
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 4096, 1, 1
__kernel void kernel_c68_sdk_401(__global float* restrict  X_T1168, __global float* restrict  X_T1175, __global const float* restrict  in1, __global const float* restrict  X_T1165, __global const float* restrict  X_I_453, __global const float* restrict  X_I_452)
{
  int tid = get_local_id(0);
  float agg[8] = {0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[6304];
  int c_gid = (get_group_id(0) * 32);
  for (int k0_gid = 0; k0_gid < 2; k0_gid += 2)
  {
    for (int k1_gid = 0; k1_gid < 2; k1_gid += 2)
    {
      {
        int gbase = ((c_gid + (k1_gid * 512)) + (k0_gid * 7168));
        int c_tid = (tid % 32);
        int k1_x1_k0_x0_tid = ((tid / 32) % 8);
        for (int k1_x1_k0_x0_lid = 0; k1_x1_k0_x0_lid < 25; k1_x1_k0_x0_lid += 1)
        {
          int k1_x1_k0_x0_cond = ((k1_x1_k0_x0_lid < 24) || (k1_x1_k0_x0_tid < 4));
          if (k1_x1_k0_x0_cond)
          {
            int k1_x1_k0_x0 = ((8 * k1_x1_k0_x0_lid) + k1_x1_k0_x0_tid);
            int lidx = ((197 * c_tid) + k1_x1_k0_x0);
            int gidx = ((gbase + c_tid) + (512 * k1_x1_k0_x0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)100351)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k0_lid = 0; k0_lid < 2; k0_lid += 1)
      {
        for (int k1_lid = 0; k1_lid < 2; k1_lid += 1)
        {
          int c_tid = (tid % 32);
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
              float val1 = in1_shared[(((((197 * c_tid) + k1_lid) + (2 * x1)) + (14 * k0_lid)) + (28 * x0))];
              int agg_idx = (x1_lid + (x0_lid * 2));
              float agg_rhs = (agg[agg_idx] + val1);
              agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)(x1_cond && x0_cond));
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
          float LX_T1167 = agg[(x1_lid + (x0_lid * 2))];
          int gout_idx = (((c_gid + c_tid) + (3584 * x0)) + (512 * x1));
          float LX_T1165 = X_T1165[gout_idx];
          float LX_I_453 = X_I_453[(c_gid + c_tid)];
          float LX_I_452 = X_I_452[(c_gid + c_tid)];
          float LX_T1168 = (LX_T1165 / LX_T1167);
          float LX_T1174 = (LX_T1168 - LX_I_453);
          float LX_T1175 = (LX_T1174 * LX_I_452);
          X_T1168[gout_idx] = LX_T1168;
          X_T1175[gout_idx] = LX_T1175;
        }
      }
    }
  }
}
