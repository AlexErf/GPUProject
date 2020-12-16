#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 4 1
// lid: 256 1 1
// Original:
// X_T255[n, x0, x1, c : _T278, _T279, _T280, _T281] = +(X_T254[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2
// With Index Variables Made Integral:
// X_T255[n, x0, x1, c : _T278, _T279, _T280, _T281] = +(X_T254[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 2, 0 <= k1 < 2, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + 2*x0 < 56, 0 <= k1 + 2*x1 < 56, 0 <= c < 128, 0 <= c < 128, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 2, 0 <= k1 < 2, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + 2*x0 < 56, 0 <= k1 + 2*x1 < 56, 0 <= c < 128 }
// Defracted:
// X_T255[n, x0, x1, c : _T278, _T279, _T280, _T281] = +(X_T254[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T255    X_T254  
//        c       128         1         1  
//       k0         2         0      7168  
//       k1         2         0       128  
//       x0        28      3584     14336  
//       x1        28       128       256  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 128, 2, 2, 28, 28 }
// Out stride: { 1, 0, 0, 3584, 128 }
// Input 1 offset: 0
// Input 1 stride: { 1, 7168, 128, 14336, 256 }
// Elementwise input X_T252 shape: fp32(1, 28, 28, 128):(100352, 3584, 128, 1):392 KiB
// Elementwise input X_I_91 shape: fp32(128):(1):512 bytes
// Elementwise input X_I_90 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(pool2_pool)]] X_T256 = div(X_T252, X_T255)
// Elementwise op: [[pid(Sub)]] X_T262 = sub(X_T256, X_I_91)
// Elementwise op: [[pid(Mul)]] X_T263 = mul(X_T262, X_I_90)
// Tile size: { 32, 2, 2, 28, 1 }
// Contraction output var shape: fp32(1, 28, 28, 128):(100352, 3584, 128, 1):392 KiB
// Computed true ops: 2007040
// Computed work groups: 112
// Computed inner loops: 1
// Computed shared mem: 14600
// Computed out regs: 4096
// Computed mem read: 14672
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 4, 1
__kernel void kernel_c124_sdk_65(__global float* restrict  X_T256, __global float* restrict  X_T263, __global const float* restrict  in1, __global const float* restrict  X_T252, __global const float* restrict  X_I_91, __global const float* restrict  X_I_90)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[3650];
  int c_gid = (get_group_id(1) * 32);
  int x1_gid = get_group_id(0);
  for (int k0_gid = 0; k0_gid < 2; k0_gid += 2)
  {
    for (int k1_gid = 0; k1_gid < 2; k1_gid += 2)
    {
      {
        int gbase = (((c_gid + (k1_gid * 128)) + (x1_gid * 256)) + (k0_gid * 7168));
        int c_tid = (tid % 32);
        int k1_x1_tid = ((tid / 32) % 2);
        int k0_x0_tid = ((tid / 64) % 4);
        for (int k0_x0_lid = 0; k0_x0_lid < 14; k0_x0_lid += 1)
        {
          int k0_x0 = ((4 * k0_x0_lid) + k0_x0_tid);
          int lidx = (((57 * c_tid) + (1825 * k1_x1_tid)) + k0_x0);
          int gidx = (((gbase + c_tid) + (128 * k1_x1_tid)) + (7168 * k0_x0));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)401407)];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k0_lid = 0; k0_lid < 2; k0_lid += 1)
      {
        for (int k1_lid = 0; k1_lid < 2; k1_lid += 1)
        {
          int c_tid = (tid % 32);
          int x0_tid = ((tid / 32) % 8);
          for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
          {
            int x0_cond = ((x0_lid < 3) || (x0_tid < 4));
            int x0 = select((int)0, (int)((8 * x0_lid) + x0_tid), (int)x0_cond);
            float val1 = in1_shared[((((57 * c_tid) + (1825 * k1_lid)) + k0_lid) + (2 * x0))];
            float agg_rhs = (agg[x0_lid] + val1);
            agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)x0_cond);
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 32);
  int x0_tid = ((tid / 32) % 8);
  for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
  {
    int x0_cond = ((x0_lid < 3) || (x0_tid < 4));
    if (x0_cond)
    {
      int x0 = ((8 * x0_lid) + x0_tid);
      float LX_T255 = agg[x0_lid];
      int gout_idx = (((c_gid + c_tid) + (3584 * x0)) + (128 * x1_gid));
      float LX_T252 = X_T252[gout_idx];
      float LX_I_91 = X_I_91[(c_gid + c_tid)];
      float LX_I_90 = X_I_90[(c_gid + c_tid)];
      float LX_T256 = (LX_T252 / LX_T255);
      float LX_T262 = (LX_T256 - LX_I_91);
      float LX_T263 = (LX_T262 * LX_I_90);
      X_T256[gout_idx] = LX_T256;
      X_T263[gout_idx] = LX_T263;
    }
  }
}
