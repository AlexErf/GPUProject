#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 4 1
// lid: 256 1 1
// Original:
// X_T328[n, x0, x1, co : _T380, _T381, _T382, _T383] = +(X_T327[n, k0 + 2*x0, k1 + 2*x1, ci] * X_I_136[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T328[n, x0, x1, co : _T380, _T381, _T382, _T383] = +(X_T327[n, k0 + 2*x0, k1 + 2*x1, ci] * X_I_136[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + 2*x0 < 28, 0 <= k1 + 2*x1 < 28, 0 <= co < 256, 0 <= co < 256, 0 <= ci < 512, 0 <= ci < 512, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + 2*x0 < 28, 0 <= k1 + 2*x1 < 28, 0 <= co < 256, 0 <= ci < 512 }
// Defracted:
// X_T328[n, x0, x1, co : _T380, _T381, _T382, _T383] = +(X_T327[n, k0 + 2*x0, k1 + 2*x1, ci] * X_I_136[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T328    X_T327   X_I_136  
//       ci       512         0         1       256  
//       co       256         1         0         1  
//       x0        14      3584     28672         0  
//       x1        14       256      1024         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 512, 256, 14, 14 }
// Out stride: { 0, 1, 3584, 256 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 28672, 1024 }
// Input 2 offset: 0
// Input 2 stride: { 256, 1, 0, 0 }
// Elementwise input X_I_135 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_134 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_133 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(Add)]] X_T329 = add(X_T328, X_I_135)
// Elementwise op: [[pid(Sub)]] X_T330 = sub(X_T329, X_I_134)
// Elementwise op: [[pid(Mul)]] X_T331 = mul(X_T330, X_I_133)
// Tile size: { 64, 64, 14, 4 }
// Contraction output var shape: fp32(1, 14, 14, 256):(50176, 3584, 256, 1):196 KiB
// Computed true ops: 128450560
// Computed work groups: 16
// Computed inner loops: 8
// Computed shared mem: 31216
// Computed out regs: 14336
// Computed mem read: 32064
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1024, 4, 1
__kernel void kernel_c29_sdk_75(__global float* restrict  X_T331, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_135, __global const float* restrict  X_I_134, __global const float* restrict  X_I_133)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3644];
  __local float in2_shared[4160];
  int co_gid = (get_group_id(0) * 64);
  int x1_gid = (get_group_id(1) * 4);
  for (int ci_gid = 0; ci_gid < 512; ci_gid += 64)
  {
    {
      int gbase = (ci_gid + (x1_gid * 1024));
      int ci_tid = (tid % 64);
      int x1_tid = ((tid / 64) % 4);
      for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
      {
        int lidx = ((ci_tid + (911 * x1_tid)) + (65 * x0_lid));
        int gidx = (((gbase + ci_tid) + (1024 * x1_tid)) + (28672 * x0_lid));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)401407)];
      }
    }
    {
      int gbase = (co_gid + (ci_gid * 256));
      int co_tid = (tid % 64);
      int ci_tid = ((tid / 64) % 4);
      for (int ci_lid = 0; ci_lid < 16; ci_lid += 1)
      {
        int ci = ((4 * ci_lid) + ci_tid);
        int lidx = (co_tid + (65 * ci));
        int gidx = ((gbase + co_tid) + (256 * ci));
        in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)131071)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 64; ci_lid += 1)
    {
      int co_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 4);
      int x0_tid = ((tid / 128) % 2);
      for (int co_lid = 0; co_lid < 2; co_lid += 1)
      {
        int co = ((32 * co_lid) + co_tid);
        for (int x0_lid = 0; x0_lid < 7; x0_lid += 1)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float val1 = in1_shared[((ci_lid + (911 * x1_tid)) + (65 * x0))];
          float val2 = in2_shared[(co + (65 * ci_lid))];
          int agg_idx = (co_lid + (x0_lid * 2));
          float agg_rhs = mad(val2, val1, agg[agg_idx]);
          agg[agg_idx] = agg_rhs;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int co_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 4);
  int x0_tid = ((tid / 128) % 2);
  int x1_cond = ((x1_gid != 12) || (x1_tid < 2));
  if (x1_cond)
  {
    for (int co_lid = 0; co_lid < 2; co_lid += 1)
    {
      int co = ((32 * co_lid) + co_tid);
      for (int x0_lid = 0; x0_lid < 7; x0_lid += 1)
      {
        int x0 = ((2 * x0_lid) + x0_tid);
        float LX_T328 = agg[(co_lid + (x0_lid * 2))];
        int gout_idx = (((co_gid + co) + (3584 * x0)) + (256 * (x1_gid + x1_tid)));
        float LX_I_135 = X_I_135[(co_gid + co)];
        float LX_I_134 = X_I_134[(co_gid + co)];
        float LX_I_133 = X_I_133[(co_gid + co)];
        float LX_T329 = (LX_T328 + LX_I_135);
        float LX_T330 = (LX_T329 - LX_I_134);
        float LX_T331 = (LX_T330 * LX_I_133);
        X_T331[gout_idx] = LX_T331;
      }
    }
  }
}
