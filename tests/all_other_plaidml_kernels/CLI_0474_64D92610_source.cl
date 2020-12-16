#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5120 1 1
// lid: 256 1 1
// Original:
// X_T1387[n, x0, x1, c : _T1968, _T1969, _T1970, _T1971] = +(X_T1386[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2
// With Index Variables Made Integral:
// X_T1387[n, x0, x1, c : _T1968, _T1969, _T1970, _T1971] = +(X_T1386[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 2, 0 <= k1 < 2, 0 <= x0 < 7, 0 <= x1 < 7, 0 <= k0 + 2*x0 < 14, 0 <= k1 + 2*x1 < 14, 0 <= c < 640, 0 <= c < 640, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 2, 0 <= k1 < 2, 0 <= x0 < 7, 0 <= x1 < 7, 0 <= k0 + 2*x0 < 14, 0 <= k1 + 2*x1 < 14, 0 <= c < 640 }
// Defracted:
// X_T1387[n, x0, x1, c : _T1968, _T1969, _T1970, _T1971] = +(X_T1386[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T1387   X_T1386  
//        c       640         1         1  
//       k0         2         0      8960  
//       k1         2         0       640  
//       x0         7      4480     17920  
//       x1         7       640      1280  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 640, 2, 2, 7, 7 }
// Out stride: { 1, 0, 0, 4480, 640 }
// Input 1 offset: 0
// Input 1 stride: { 1, 8960, 640, 17920, 1280 }
// Elementwise input X_T1385 shape: fp32(1, 7, 7, 640):(31360, 4480, 640, 1):122.5 KiB
// Elementwise input X_I_533 shape: fp32(640):(1):2.5 KiB
// Elementwise input X_I_532 shape: fp32(640):(1):2.5 KiB
// Elementwise op: [[pid(pool4_pool)]] X_T1388 = div(X_T1385, X_T1387)
// Elementwise op: [[pid(Sub)]] X_T1394 = sub(X_T1388, X_I_533)
// Elementwise op: [[pid(Mul)]] X_T1395 = mul(X_T1394, X_I_532)
// Tile size: { 32, 2, 2, 7, 7 }
// Contraction output var shape: fp32(1, 7, 7, 640):(31360, 4480, 640, 1):122.5 KiB
// Computed true ops: 627200
// Computed work groups: 20
// Computed inner loops: 1
// Computed shared mem: 25216
// Computed out regs: 8192
// Computed mem read: 25676
// Computed mem write: 12544
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5120, 1, 1
__kernel void kernel_c108_sdk_473(__global float* restrict  X_T1388, __global float* restrict  X_T1395, __global const float* restrict  in1, __global const float* restrict  X_T1385, __global const float* restrict  X_I_533, __global const float* restrict  X_I_532)
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
        int gbase = ((c_gid + (k1_gid * 640)) + (k0_gid * 8960));
        int c_tid = (tid % 32);
        int k1_x1_k0_x0_tid = ((tid / 32) % 8);
        for (int k1_x1_k0_x0_lid = 0; k1_x1_k0_x0_lid < 25; k1_x1_k0_x0_lid += 1)
        {
          int k1_x1_k0_x0_cond = ((k1_x1_k0_x0_lid < 24) || (k1_x1_k0_x0_tid < 4));
          if (k1_x1_k0_x0_cond)
          {
            int k1_x1_k0_x0 = ((8 * k1_x1_k0_x0_lid) + k1_x1_k0_x0_tid);
            int lidx = ((197 * c_tid) + k1_x1_k0_x0);
            int gidx = ((gbase + c_tid) + (640 * k1_x1_k0_x0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)125439)];
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
          float LX_T1387 = agg[(x1_lid + (x0_lid * 2))];
          int gout_idx = (((c_gid + c_tid) + (4480 * x0)) + (640 * x1));
          float LX_T1385 = X_T1385[gout_idx];
          float LX_I_533 = X_I_533[(c_gid + c_tid)];
          float LX_I_532 = X_I_532[(c_gid + c_tid)];
          float LX_T1388 = (LX_T1385 / LX_T1387);
          float LX_T1394 = (LX_T1388 - LX_I_533);
          float LX_T1395 = (LX_T1394 * LX_I_532);
          X_T1388[gout_idx] = LX_T1388;
          X_T1395[gout_idx] = LX_T1395;
        }
      }
    }
  }
}
