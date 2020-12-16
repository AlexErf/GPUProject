#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2304 18 1
// lid: 256 1 1
// Original:
// X_T324[n, x0, x1, c : _T437, _T438, _T439, _T440] = +(X_T323[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T324[n, x0, x1, c : _T437, _T438, _T439, _T440] = +(X_T323[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 35, 0 <= x1 < 35, 0 <= -1 + k0 + x0 < 35, 0 <= -1 + k1 + x1 < 35, 0 <= c < 288, 0 <= c < 288, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 35, 0 <= x1 < 35, 0 <= -1 + k0 + x0 < 35, 0 <= -1 + k1 + x1 < 35, 0 <= c < 288 }
// Defracted:
// X_T324[n, x0, x1, c : _T437, _T438, _T439, _T440] = +(X_T323[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T324    X_T323  
//        c       288         1         1  
//       k0         3         0     10080  
//       k1         3         0       288  
//       x0        35     10080     10080  
//       x1        35       288       288  
//      off                   0    -10368  
//      vec                   1         1  
// Constraint: (0,-1,0,-1,0) <= -1
// Constraint: (0,1,0,1,0) <= 35
// Constraint: (0,0,-1,0,-1) <= -1
// Constraint: (0,0,1,0,1) <= 35
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 288, 3, 3, 35, 35 }
// Out stride: { 1, 0, 0, 10080, 288 }
// Input 1 offset: -10368
// Input 1 stride: { 1, 10080, 288, 10080, 288 }
// Elementwise input X_T322 shape: fp32(1, 35, 35, 288):(352800, 10080, 288, 1):1378.12 KiB
// Elementwise op: [[pid(average_pooling2d_3)]] X_T325 = div(X_T322, X_T324)
// Tile size: { 32, 3, 3, 35, 2 }
// Contraction output var shape: fp32(1, 35, 35, 288):(352800, 10080, 288, 1):1378.12 KiB
// Computed true ops: 9525600
// Computed work groups: 162
// Computed inner loops: 1
// Computed shared mem: 18960
// Computed out regs: 9216
// Computed mem read: 19224
// Computed mem write: 8960
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2304, 18, 1
__kernel void kernel_c56_sdk_104(__global float* restrict  X_T325, __global const float* restrict  in1, __global const float* restrict  X_T322)
{
  in1 = (in1 + -10368);
  int tid = get_local_id(0);
  float agg[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[4740];
  int c_gid = (get_group_id(0) * 32);
  int x1_gid = (get_group_id(1) * 2);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = (((c_gid + (k1_gid * 288)) + (x1_gid * 288)) + (k0_gid * 10080));
        int c_tid = (tid % 32);
        int k1_x1_tid = ((tid / 32) % 4);
        int k0_x0_tid = ((tid / 128) % 2);
        for (int k0_x0_lid = 0; k0_x0_lid < 19; k0_x0_lid += 1)
        {
          int k0_x0_cond = ((k0_x0_lid < 18) || (k0_x0_tid < 1));
          if (k0_x0_cond)
          {
            int k0_x0 = ((2 * k0_x0_lid) + k0_x0_tid);
            int lidx = (((37 * c_tid) + (1185 * k1_x1_tid)) + k0_x0);
            int gidx = (((gbase + c_tid) + (288 * k1_x1_tid)) + (10080 * k0_x0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)10368, (int)363167)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if ((((((-1 * k0_gid) <= -1) && ((((k0_gid + 3) - 1) + 34) <= 35)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -1)) && ((((k1_gid + 3) - 1) + ((x1_gid + 2) - 1)) <= 35)))
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int c_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 2);
            int x0_tid = ((tid / 64) % 4);
            for (int x0_lid = 0; x0_lid < 9; x0_lid += 1)
            {
              int x0_cond = ((x0_lid < 8) || (x0_tid < 3));
              int x0 = select((int)0, (int)((4 * x0_lid) + x0_tid), (int)x0_cond);
              float val1 = in1_shared[(((((37 * c_tid) + (1185 * k1_lid)) + (1185 * x1_tid)) + k0_lid) + x0)];
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
            int x1_tid = ((tid / 32) % 2);
            int x0_tid = ((tid / 64) % 4);
            for (int x0_lid = 0; x0_lid < 9; x0_lid += 1)
            {
              int x0_cond = ((x0_lid < 8) || (x0_tid < 3));
              int x0 = select((int)0, (int)((4 * x0_lid) + x0_tid), (int)x0_cond);
              float val1 = in1_shared[(((((37 * c_tid) + (1185 * k1_lid)) + (1185 * x1_tid)) + k0_lid) + x0)];
              float agg_rhs = (agg[x0_lid] + val1);
              agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)(x0_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * x0)) <= -1) && (((k0_gid + k0_lid) + x0) <= 35)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1_tid))) <= -1)) && (((k1_gid + k1_lid) + (x1_gid + x1_tid)) <= 35))));
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
  int x1_cond = ((x1_gid != 34) || (x1_tid < 1));
  if (x1_cond)
  {
    for (int x0_lid = 0; x0_lid < 9; x0_lid += 1)
    {
      int x0_cond = ((x0_lid < 8) || (x0_tid < 3));
      if (x0_cond)
      {
        int x0 = ((4 * x0_lid) + x0_tid);
        float LX_T324 = agg[x0_lid];
        int gout_idx = (((c_gid + c_tid) + (10080 * x0)) + (288 * (x1_gid + x1_tid)));
        if (((gout_idx >= 0) && (gout_idx < 352800)))
        {
          float LX_T322 = X_T322[gout_idx];
          float LX_T325 = (LX_T322 / LX_T324);
          X_T325[gout_idx] = LX_T325;
        }
      }
    }
  }
}
