#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Original:
// X_T104[n, x0, x1, co : _T59, _T60, _T61, _T62] = +(X_T103[n, -1 + k0 + x0, -1 + k1 + x1, ci] * X_I_23[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T104[n, x0, x1, co : _T59, _T60, _T61, _T62] = +(X_T103[n, -1 + k0 + x0, -1 + k1 + x1, ci] * X_I_23[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= co < 32, 0 <= co < 32, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= -1 + k0 + x0 < 56, 0 <= -1 + k1 + x1 < 56, 0 <= ci < 128, 0 <= ci < 128, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= co < 32, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= -1 + k0 + x0 < 56, 0 <= -1 + k1 + x1 < 56, 0 <= ci < 128 }
// Defracted:
// X_T104[n, x0, x1, co : _T59, _T60, _T61, _T62] = +(X_T103[n, -1 + k0 + x0, -1 + k1 + x1, ci] * X_I_23[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T104    X_T103    X_I_23  
//       ci       128         0         1        32  
//       co        32         1         0         1  
//       k0         3         0      7168     12288  
//       k1         3         0       128      4096  
//       x0        56      1792      7168         0  
//       x1        56        32       128         0  
//      off                   0     -7296         0  
//      vec                   1         1         1  
// Constraint: (0,0,-1,0,-1,0) <= -1
// Constraint: (0,0,1,0,1,0) <= 56
// Constraint: (0,0,0,-1,0,-1) <= -1
// Constraint: (0,0,0,1,0,1) <= 56
// 
// Names: { ci, co, k0, k1, x0, x1 }
// Ranges: { 128, 32, 3, 3, 56, 56 }
// Out stride: { 0, 1, 0, 0, 1792, 32 }
// Input 1 offset: -7296
// Input 1 stride: { 1, 0, 7168, 128, 7168, 128 }
// Input 2 offset: 0
// Input 2 stride: { 32, 1, 12288, 4096, 0, 0 }
// Tile size: { 32, 32, 3, 1, 56, 2 }
// Contraction output var shape: fp32(1, 56, 56, 32):(100352, 1792, 32, 1):392 KiB
// Computed true ops: 231211008
// Computed work groups: 28
// Computed inner loops: 12
// Computed shared mem: 27400
// Computed out regs: 14336
// Computed mem read: 27136
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c108_sdk_12(__global float* restrict  X_T104, __global const float* restrict  in1, __global const float* restrict  in2)
{
  in1 = (in1 + -7296);
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3778];
  __local float in2_shared[3072];
  int x1_gid = (get_group_id(0) * 2);
  for (int ci_gid = 0; ci_gid < 128; ci_gid += 32)
  {
    for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
    {
      for (int k1_gid = 0; k1_gid < 3; k1_gid += 1)
      {
        {
          int gbase = (((ci_gid + (k1_gid * 128)) + (x1_gid * 128)) + (k0_gid * 7168));
          int ci_tid = (tid % 32);
          int k1_x1_tid = ((tid / 32) % 2);
          int k0_x0_tid = ((tid / 64) % 4);
          for (int k0_x0_lid = 0; k0_x0_lid < 15; k0_x0_lid += 1)
          {
            int k0_x0_cond = ((k0_x0_lid < 14) || (k0_x0_tid < 2));
            if (k0_x0_cond)
            {
              int k0_x0 = ((4 * k0_x0_lid) + k0_x0_tid);
              int lidx = (((59 * ci_tid) + (1889 * k1_x1_tid)) + k0_x0);
              int gidx = (((gbase + ci_tid) + (128 * k1_x1_tid)) + (7168 * k0_x0));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)7296, (int)408703)];
            }
          }
        }
        {
          int gbase = (((ci_gid * 32) + (k1_gid * 4096)) + (k0_gid * 12288));
          int co_ci_tid = (tid % 256);
          for (int co_ci_lid = 0; co_ci_lid < 4; co_ci_lid += 1)
          {
            int co_ci = ((256 * co_ci_lid) + co_ci_tid);
            for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
            {
              int lidx = ((3 * co_ci) + k0_lid);
              int gidx = ((gbase + co_ci) + (12288 * k0_lid));
              in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)36863)];
            }
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((((((-1 * k0_gid) <= -1) && ((((k0_gid + 3) - 1) + 55) <= 56)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -1)) && ((((k1_gid + 1) - 1) + ((x1_gid + 2) - 1)) <= 56)))
        {
          for (int ci_lid = 0; ci_lid < 32; ci_lid += 1)
          {
            for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
            {
              int co_tid = (tid % 32);
              int x1_tid = ((tid / 32) % 2);
              int x0_tid = ((tid / 64) % 4);
              for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
              {
                int x0 = ((4 * x0_lid) + x0_tid);
                float val1 = in1_shared[((((59 * ci_lid) + (1889 * x1_tid)) + k0_lid) + x0)];
                float val2 = in2_shared[(((3 * co_tid) + (96 * ci_lid)) + k0_lid)];
                float agg_rhs = mad(val2, val1, agg[x0_lid]);
                agg[x0_lid] = agg_rhs;
              }
            }
          }
        }
        else
        {
          for (int ci_lid = 0; ci_lid < 32; ci_lid += 1)
          {
            for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
            {
              int co_tid = (tid % 32);
              int x1_tid = ((tid / 32) % 2);
              int x0_tid = ((tid / 64) % 4);
              for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
              {
                int x0 = ((4 * x0_lid) + x0_tid);
                float val1 = in1_shared[((((59 * ci_lid) + (1889 * x1_tid)) + k0_lid) + x0)];
                float val2 = in2_shared[(((3 * co_tid) + (96 * ci_lid)) + k0_lid)];
                float agg_rhs = mad(val2, val1, agg[x0_lid]);
                agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)((((((-1 * (k0_gid + k0_lid)) + (-1 * x0)) <= -1) && (((k0_gid + k0_lid) + x0) <= 56)) && (((-1 * k1_gid) + (-1 * (x1_gid + x1_tid))) <= -1)) && ((k1_gid + (x1_gid + x1_tid)) <= 56)));
              }
            }
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }
  }
  int co_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 2);
  int x0_tid = ((tid / 64) % 4);
  for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
  {
    int x0 = ((4 * x0_lid) + x0_tid);
    float LX_T104 = agg[x0_lid];
    int gout_idx = ((co_tid + (1792 * x0)) + (32 * (x1_gid + x1_tid)));
    if (((gout_idx >= 0) && (gout_idx < 100352)))
    {
      X_T104[gout_idx] = LX_T104;
    }
  }
}
