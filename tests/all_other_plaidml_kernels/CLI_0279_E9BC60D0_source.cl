#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 2 1
// lid: 256 1 1
// Original:
// X_T2546[n, x0, x1, co : _T4050, _T4051, _T4052, _T4053] = +(X_T2545[n, k0 + x0, k1 + x1, ci] * X_I_868[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T2546[n, x0, x1, co : _T4050, _T4051, _T4052, _T4053] = +(X_T2545[n, k0 + x0, k1 + x1, ci] * X_I_868[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 7, 0 <= x1 < 7, 0 <= k0 + x0 < 7, 0 <= k1 + x1 < 7, 0 <= co < 176, 0 <= co < 176, 0 <= ci < 1056, 0 <= ci < 1056, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 7, 0 <= x1 < 7, 0 <= k0 + x0 < 7, 0 <= k1 + x1 < 7, 0 <= co < 176, 0 <= ci < 1056 }
// Defracted:
// X_T2546[n, x0, x1, co : _T4050, _T4051, _T4052, _T4053] = +(X_T2545[n, k0 + x0, k1 + x1, ci] * X_I_868[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T2546   X_T2545   X_I_868  
//       ci      1056         0         1       176  
//       co       176         1         0         1  
//       x0         7      1232      7392         0  
//       x1         7       176      1056         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 1056, 176, 7, 7 }
// Out stride: { 0, 1, 1232, 176 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 7392, 1056 }
// Input 2 offset: 0
// Input 2 stride: { 176, 1, 0, 0 }
// Elementwise input X_I_867 shape: fp32(176):(1):704 bytes
// Elementwise input X_I_866 shape: fp32(176):(1):704 bytes
// Elementwise op: [[pid(Sub)]] X_T2547 = sub(X_T2546, X_I_867)
// Elementwise op: [[pid(Mul)]] X_T2548 = mul(X_T2547, X_I_866)
// Tile size: { 32, 32, 7, 4 }
// Contraction output var shape: fp32(1, 7, 7, 176):(8624, 1232, 176, 1):33.6875 KiB
// Computed true ops: 36427776
// Computed work groups: 12
// Computed inner loops: 33
// Computed shared mem: 7824
// Computed out regs: 4096
// Computed mem read: 7904
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 2, 1
__kernel void kernel_c42_sdk_979(__global float* restrict  X_T2548, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_867, __global const float* restrict  X_I_866)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[900];
  __local float in2_shared[1056];
  int co_gid = (get_group_id(0) * 32);
  int x1_gid = (get_group_id(1) * 4);
  for (int ci_gid = 0; ci_gid < 1056; ci_gid += 32)
  {
    {
      int gbase = (ci_gid + (x1_gid * 1056));
      int ci_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 4);
      int x0_tid = ((tid / 128) % 2);
      for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 3) || (x0_tid < 1));
        if (x0_cond)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          int lidx = (((7 * ci_tid) + (225 * x1_tid)) + x0);
          int gidx = (((gbase + ci_tid) + (1056 * x1_tid)) + (7392 * x0));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)51743)];
        }
      }
    }
    {
      int gbase = (co_gid + (ci_gid * 176));
      int co_tid = (tid % 32);
      int ci_tid = ((tid / 32) % 8);
      for (int ci_lid = 0; ci_lid < 4; ci_lid += 1)
      {
        int ci = ((8 * ci_lid) + ci_tid);
        int lidx = (co_tid + (33 * ci));
        int gidx = ((gbase + co_tid) + (176 * ci));
        in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)185855)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 32; ci_lid += 1)
    {
      int co_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 4);
      int x0_tid = ((tid / 128) % 2);
      for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 3) || (x0_tid < 1));
        int x0 = select((int)0, (int)((2 * x0_lid) + x0_tid), (int)x0_cond);
        float val1 = in1_shared[(((7 * ci_lid) + (225 * x1_tid)) + x0)];
        float val2 = in2_shared[(co_tid + (33 * ci_lid))];
        float agg_rhs = mad(val2, val1, agg[x0_lid]);
        agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)x0_cond);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int co_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 4);
  int x0_tid = ((tid / 128) % 2);
  int co_cond = ((co_gid != 160) || (co_tid < 16));
  if (co_cond)
  {
    int x1_cond = ((x1_gid != 4) || (x1_tid < 3));
    if (x1_cond)
    {
      for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 3) || (x0_tid < 1));
        if (x0_cond)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float LX_T2546 = agg[x0_lid];
          int gout_idx = (((co_gid + co_tid) + (1232 * x0)) + (176 * (x1_gid + x1_tid)));
          float LX_I_867 = X_I_867[(co_gid + co_tid)];
          float LX_I_866 = X_I_866[(co_gid + co_tid)];
          float LX_T2547 = (LX_T2546 - LX_I_867);
          float LX_T2548 = (LX_T2547 * LX_I_866);
          X_T2548[gout_idx] = LX_T2548;
        }
      }
    }
  }
}
