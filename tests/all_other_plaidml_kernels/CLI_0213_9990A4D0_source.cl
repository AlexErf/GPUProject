#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2560 2 1
// lid: 256 1 1
// Original:
// X_T994[n, x0, x1, c : _T1412, _T1413, _T1414, _T1415] = +(X_T993[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T994[n, x0, x1, c : _T1412, _T1413, _T1414, _T1415] = +(X_T993[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 8, 0 <= x1 < 8, 0 <= -1 + k0 + x0 < 8, 0 <= -1 + k1 + x1 < 8, 0 <= c < 1280, 0 <= c < 1280, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 8, 0 <= x1 < 8, 0 <= -1 + k0 + x0 < 8, 0 <= -1 + k1 + x1 < 8, 0 <= c < 1280 }
// Defracted:
// X_T994[n, x0, x1, c : _T1412, _T1413, _T1414, _T1415] = +(X_T993[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T994    X_T993  
//        c      1280         1         1  
//       k0         3         0     10240  
//       k1         3         0      1280  
//       x0         8     10240     10240  
//       x1         8      1280      1280  
//      off                   0    -11520  
//      vec                   1         1  
// Constraint: (0,-1,0,-1,0) <= -1
// Constraint: (0,1,0,1,0) <= 8
// Constraint: (0,0,-1,0,-1) <= -1
// Constraint: (0,0,1,0,1) <= 8
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 1280, 3, 3, 8, 8 }
// Out stride: { 1, 0, 0, 10240, 1280 }
// Input 1 offset: -11520
// Input 1 stride: { 1, 10240, 1280, 10240, 1280 }
// Elementwise input X_T992 shape: fp32(1, 8, 8, 1280):(81920, 10240, 1280, 1):320 KiB
// Elementwise op: [[pid(average_pooling2d_8)]] X_T995 = div(X_T992, X_T994)
// Tile size: { 128, 3, 3, 4, 8 }
// Contraction output var shape: fp32(1, 8, 8, 1280):(81920, 10240, 1280, 1):320 KiB
// Computed true ops: 2211840
// Computed work groups: 20
// Computed inner loops: 1
// Computed shared mem: 25800
// Computed out regs: 16384
// Computed mem read: 26112
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2560, 2, 1
__kernel void kernel_c56_sdk_343(__global float* restrict  X_T995, __global const float* restrict  in1, __global const float* restrict  X_T992)
{
  in1 = (in1 + -11520);
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[6450];
  int c_gid = (get_group_id(0) * 128);
  int x0_gid = (get_group_id(1) * 4);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = (((c_gid + (k1_gid * 1280)) + (k0_gid * 10240)) + (x0_gid * 10240));
        int c_tid = (tid % 128);
        int k1_x1_k0_x0_tid = ((tid / 128) % 2);
        for (int k1_x1_k0_x0_lid = 0; k1_x1_k0_x0_lid < 25; k1_x1_k0_x0_lid += 1)
        {
          int k1_x1_k0_x0 = ((2 * k1_x1_k0_x0_lid) + k1_x1_k0_x0_tid);
          int lidx = (c_tid + (129 * k1_x1_k0_x0));
          int gidx = ((gbase + c_tid) + (1280 * k1_x1_k0_x0));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)11520, (int)93439)];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -1) && ((((k0_gid + 3) - 1) + ((x0_gid + 4) - 1)) <= 8)) && ((-1 * k1_gid) <= -1)) && ((((k1_gid + 3) - 1) + 7) <= 8)))
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int c_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int c_lid = 0; c_lid < 4; c_lid += 1)
            {
              int c = ((32 * c_lid) + c_tid);
              for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
              {
                int x1 = ((4 * x1_lid) + x1_tid);
                for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
                {
                  int x0 = ((2 * x0_lid) + x0_tid);
                  float val1 = in1_shared[((((c + (129 * k1_lid)) + (129 * x1)) + (1032 * k0_lid)) + (1032 * x0))];
                  int agg_idx = ((c_lid + (x1_lid * 4)) + (x0_lid * 8));
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
            for (int c_lid = 0; c_lid < 4; c_lid += 1)
            {
              int c = ((32 * c_lid) + c_tid);
              for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
              {
                int x1 = ((4 * x1_lid) + x1_tid);
                for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
                {
                  int x0 = ((2 * x0_lid) + x0_tid);
                  float val1 = in1_shared[((((c + (129 * k1_lid)) + (129 * x1)) + (1032 * k0_lid)) + (1032 * x0))];
                  int agg_idx = ((c_lid + (x1_lid * 4)) + (x0_lid * 8));
                  float agg_rhs = (agg[agg_idx] + val1);
                  agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)((((((-1 * (k0_gid + k0_lid)) + (-1 * (x0_gid + x0))) <= -1) && (((k0_gid + k0_lid) + (x0_gid + x0)) <= 8)) && (((-1 * (k1_gid + k1_lid)) + (-1 * x1)) <= -1)) && (((k1_gid + k1_lid) + x1) <= 8)));
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
  for (int c_lid = 0; c_lid < 4; c_lid += 1)
  {
    int c = ((32 * c_lid) + c_tid);
    for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
      {
        int x0 = ((2 * x0_lid) + x0_tid);
        float LX_T994 = agg[((c_lid + (x1_lid * 4)) + (x0_lid * 8))];
        int gout_idx = (((c_gid + c) + (10240 * (x0_gid + x0))) + (1280 * x1));
        if (((gout_idx >= 0) && (gout_idx < 81920)))
        {
          float LX_T992 = X_T992[gout_idx];
          float LX_T995 = (LX_T992 / LX_T994);
          X_T995[gout_idx] = LX_T995;
        }
      }
    }
  }
}
