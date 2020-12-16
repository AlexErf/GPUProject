#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Original:
// X_T420[n, x0, x1, c : _T621, _T622, _T623, _T624] = +(X_T419[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T420[n, x0, x1, c : _T621, _T622, _T623, _T624] = +(X_T419[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= c < 22, 0 <= c < 22, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= -1 + k0 + x0 < 28, 0 <= -1 + k1 + x1 < 28, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= c < 22, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= -1 + k0 + x0 < 28, 0 <= -1 + k1 + x1 < 28 }
// Defracted:
// X_T420[n, x0, x1, c : _T621, _T622, _T623, _T624] = +(X_T419[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T420    X_T419  
//        c        22         1         1  
//       k0         3         0       616  
//       k1         3         0        22  
//       x0        28       616       616  
//       x1        28        22        22  
//      off                   0      -638  
//      vec                   1         1  
// Constraint: (0,-1,0,-1,0) <= -1
// Constraint: (0,1,0,1,0) <= 28
// Constraint: (0,0,-1,0,-1) <= -1
// Constraint: (0,0,1,0,1) <= 28
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 22, 3, 3, 28, 28 }
// Out stride: { 1, 0, 0, 616, 22 }
// Input 1 offset: -638
// Input 1 stride: { 1, 616, 22, 616, 22 }
// Tile size: { 22, 3, 3, 4, 4 }
// Contraction output var shape: fp32(1, 28, 28, 22):(17248, 616, 22, 1):67.375 KiB
// Computed true ops: 310464
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 3192
// Computed out regs: 2048
// Computed mem read: 3072
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c42_sdk_143(__global float* restrict  X_T420, __global const float* restrict  in1)
{
  in1 = (in1 + -638);
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[798];
  int x1_gid = (get_group_id(0) * 4);
  int x0_gid = (get_group_id(1) * 4);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((((k1_gid * 22) + (x1_gid * 22)) + (k0_gid * 616)) + (x0_gid * 616));
        int c_k1_x1_tid = (tid % 256);
        int c_k1_x1_cond = (c_k1_x1_tid < 132);
        if (c_k1_x1_cond)
        {
          for (int k0_x0_lid = 0; k0_x0_lid < 6; k0_x0_lid += 1)
          {
            int lidx = (c_k1_x1_tid + (133 * k0_x0_lid));
            int gidx = ((gbase + c_k1_x1_tid) + (616 * k0_x0_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)638, (int)17885)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -1) && ((((k0_gid + 3) - 1) + ((x0_gid + 4) - 1)) <= 28)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -1)) && ((((k1_gid + 3) - 1) + ((x1_gid + 4) - 1)) <= 28)))
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int c_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            int c_cond = (c_tid < 22);
            int c = select((int)0, (int)c_tid, (int)c_cond);
            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
            {
              int x0 = ((2 * x0_lid) + x0_tid);
              float val1 = in1_shared[((((c + (22 * k1_lid)) + (22 * x1_tid)) + (133 * k0_lid)) + (133 * x0))];
              float agg_rhs = (agg[x0_lid] + val1);
              agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)c_cond);
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
            int c_cond = (c_tid < 22);
            int c = select((int)0, (int)c_tid, (int)c_cond);
            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
            {
              int x0 = ((2 * x0_lid) + x0_tid);
              float val1 = in1_shared[((((c + (22 * k1_lid)) + (22 * x1_tid)) + (133 * k0_lid)) + (133 * x0))];
              float agg_rhs = (agg[x0_lid] + val1);
              agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)(c_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * (x0_gid + x0))) <= -1) && (((k0_gid + k0_lid) + (x0_gid + x0)) <= 28)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1_tid))) <= -1)) && (((k1_gid + k1_lid) + (x1_gid + x1_tid)) <= 28))));
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
  int c_cond = (c_tid < 22);
  if (c_cond)
  {
    for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
    {
      int x0 = ((2 * x0_lid) + x0_tid);
      float LX_T420 = agg[x0_lid];
      int gout_idx = ((c_tid + (616 * (x0_gid + x0))) + (22 * (x1_gid + x1_tid)));
      if (((gout_idx >= 0) && (gout_idx < 17248)))
      {
        X_T420[gout_idx] = LX_T420;
      }
    }
  }
}
