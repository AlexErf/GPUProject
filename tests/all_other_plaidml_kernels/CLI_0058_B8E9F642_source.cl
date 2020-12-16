#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 4864 37 1
// lid: 256 1 1
// Original:
// X_T149[n, x0, x1, c : _T114, _T115, _T116, _T117] = >(X_T148[n, -1 + k0 + 2*x0, -1 + k1 + 2*x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T149[n, x0, x1, c : _T114, _T115, _T116, _T117] = >(X_T148[n, -1 + k0 + 2*x0, -1 + k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 74, 0 <= x1 < 74, 0 <= c < 128, 0 <= c < 128, 0 <= -1 + k0 + 2*x0 < 147, 0 <= -1 + k1 + 2*x1 < 147, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 74, 0 <= x1 < 74, 0 <= c < 128, 0 <= -1 + k0 + 2*x0 < 147, 0 <= -1 + k1 + 2*x1 < 147 }
// Defracted:
// X_T149[n, x0, x1, c : _T114, _T115, _T116, _T117] = >(X_T148[n, -1 + k0 + 2*x0, -1 + k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T149    X_T148  
//        c       128         1         1  
//       k0         3         0     18816  
//       k1         3         0       128  
//       x0        74      9472     37632  
//       x1        74       128       256  
//      off                   0    -18944  
//      vec                   1         1  
// Constraint: (0,-1,0,-2,0) <= -1
// Constraint: (0,1,0,2,0) <= 147
// Constraint: (0,0,-1,0,-2) <= -1
// Constraint: (0,0,1,0,2) <= 147
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 128, 3, 3, 74, 74 }
// Out stride: { 1, 0, 0, 9472, 128 }
// Input 1 offset: -18944
// Input 1 stride: { 1, 18816, 128, 37632, 256 }
// Tile size: { 128, 3, 3, 2, 4 }
// Contraction output var shape: fp32(1, 74, 74, 128):(700928, 9472, 128, 1):2738 KiB
// Computed true ops: 12616704
// Computed work groups: 703
// Computed inner loops: 1
// Computed shared mem: 23040
// Computed out regs: 4096
// Computed mem read: 23040
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 4864, 37, 1
__kernel void kernel_c28_sdk_48(__global float* restrict  X_T149, __global const float* restrict  in1)
{
  in1 = (in1 + -18944);
  int tid = get_local_id(0);
  float agg[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, };
  __local float in1_shared[5760];
  int x1_gid = (get_group_id(0) * 4);
  int x0_gid = (get_group_id(1) * 2);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((((k1_gid * 128) + (x1_gid * 256)) + (k0_gid * 18816)) + (x0_gid * 37632));
        int c_k1_x1_tid = (tid % 256);
        for (int c_k1_x1_lid = 0; c_k1_x1_lid < 5; c_k1_x1_lid += 1)
        {
          int c_k1_x1_cond = ((c_k1_x1_lid < 4) || (c_k1_x1_tid < 128));
          if (c_k1_x1_cond)
          {
            int c_k1_x1 = ((256 * c_k1_x1_lid) + c_k1_x1_tid);
            for (int k0_x0_lid = 0; k0_x0_lid < 5; k0_x0_lid += 1)
            {
              int lidx = ((5 * c_k1_x1) + k0_x0_lid);
              int gidx = ((gbase + c_k1_x1) + (18816 * k0_x0_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)18944, (int)2784895)];
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (((((((-1 * k0_gid) + (-2 * x0_gid)) <= -1) && ((((k0_gid + 3) - 1) + (2 * ((x0_gid + 2) - 1))) <= 147)) && (((-1 * k1_gid) + (-2 * x1_gid)) <= -1)) && ((((k1_gid + 3) - 1) + (2 * ((x1_gid + 4) - 1))) <= 147)))
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
              float val1 = in1_shared[(((((5 * c) + (640 * k1_lid)) + (1280 * x1_tid)) + k0_lid) + (2 * x0_tid))];
              float agg_rhs = select((float)agg[c_lid], (float)val1, (int)(val1 > agg[c_lid]));
              agg[c_lid] = agg_rhs;
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
              float val1 = in1_shared[(((((5 * c) + (640 * k1_lid)) + (1280 * x1_tid)) + k0_lid) + (2 * x0_tid))];
              float agg_rhs = select((float)agg[c_lid], (float)val1, (int)(val1 > agg[c_lid]));
              agg[c_lid] = select((float)agg[c_lid], (float)agg_rhs, (int)((((((-1 * (k0_gid + k0_lid)) + (-2 * (x0_gid + x0_tid))) <= -1) && (((k0_gid + k0_lid) + (2 * (x0_gid + x0_tid))) <= 147)) && (((-1 * (k1_gid + k1_lid)) + (-2 * (x1_gid + x1_tid))) <= -1)) && (((k1_gid + k1_lid) + (2 * (x1_gid + x1_tid))) <= 147)));
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
  int x1_cond = ((x1_gid != 72) || (x1_tid < 2));
  if (x1_cond)
  {
    for (int c_lid = 0; c_lid < 4; c_lid += 1)
    {
      int c = ((32 * c_lid) + c_tid);
      float LX_T149 = agg[c_lid];
      LX_T149 = select((float)LX_T149, (float)0, (int)(LX_T149 == (float)-FLT_MAX));
      int gout_idx = ((c + (9472 * (x0_gid + x0_tid))) + (128 * (x1_gid + x1_tid)));
      if (((gout_idx >= 0) && (gout_idx < 700928)))
      {
        X_T149[gout_idx] = LX_T149;
      }
    }
  }
}
