#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 4864 19 1
// lid: 256 1 1
// Original:
// X_T62[n, x0, x1, c : _T42, _T43, _T44, _T45] = >(X_T61[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T62[n, x0, x1, c : _T42, _T43, _T44, _T45] = >(X_T61[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= c < 64, 0 <= c < 64, 0 <= x0 < 73, 0 <= x1 < 73, 0 <= k0 + 2*x0 < 147, 0 <= k1 + 2*x1 < 147, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= c < 64, 0 <= x0 < 73, 0 <= x1 < 73, 0 <= k0 + 2*x0 < 147, 0 <= k1 + 2*x1 < 147 }
// Defracted:
// X_T62[n, x0, x1, c : _T42, _T43, _T44, _T45] = >(X_T61[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T62     X_T61  
//        c        64         1         1  
//       k0         3         0      9408  
//       k1         3         0        64  
//       x0        73      4672     18816  
//       x1        73        64       128  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 64, 3, 3, 73, 73 }
// Out stride: { 1, 0, 0, 4672, 64 }
// Input 1 offset: 0
// Input 1 stride: { 1, 9408, 64, 18816, 128 }
// Tile size: { 64, 3, 3, 4, 4 }
// Contraction output var shape: fp32(1, 73, 73, 64):(341056, 4672, 64, 1):1332.25 KiB
// Computed true ops: 6139008
// Computed work groups: 361
// Computed inner loops: 1
// Computed shared mem: 20736
// Computed out regs: 4096
// Computed mem read: 20736
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 4864, 19, 1
__kernel void kernel_c51_sdk_9(__global float* restrict  X_T62, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, };
  __local float in1_shared[5184];
  int x1_gid = (get_group_id(0) * 4);
  int x0_gid = (get_group_id(1) * 4);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((((k1_gid * 64) + (x1_gid * 128)) + (k0_gid * 9408)) + (x0_gid * 18816));
        int c_k1_x1_tid = (tid % 256);
        for (int c_k1_x1_lid = 0; c_k1_x1_lid < 3; c_k1_x1_lid += 1)
        {
          int c_k1_x1_cond = ((c_k1_x1_lid < 2) || (c_k1_x1_tid < 64));
          if (c_k1_x1_cond)
          {
            int c_k1_x1 = ((256 * c_k1_x1_lid) + c_k1_x1_tid);
            for (int k0_x0_lid = 0; k0_x0_lid < 9; k0_x0_lid += 1)
            {
              int lidx = ((9 * c_k1_x1) + k0_x0_lid);
              int gidx = ((gbase + c_k1_x1) + (9408 * k0_x0_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)1382975)];
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
      {
        for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
        {
          int c_tid = (tid % 32);
          int x1_tid = ((tid / 32) % 4);
          int x0_tid = ((tid / 128) % 2);
          for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
          {
            int x0 = ((2 * x0_lid) + x0_tid);
            for (int c_lid = 0; c_lid < 2; c_lid += 1)
            {
              int c = ((32 * c_lid) + c_tid);
              float val1 = in1_shared[(((((9 * c) + (576 * k1_lid)) + (1152 * x1_tid)) + k0_lid) + (2 * x0))];
              int agg_idx = (c_lid + (x0_lid * 2));
              float agg_rhs = select((float)agg[agg_idx], (float)val1, (int)(val1 > agg[agg_idx]));
              agg[agg_idx] = agg_rhs;
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
  int x1_cond = ((x1_gid != 72) || (x1_tid < 1));
  if (x1_cond)
  {
    for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
    {
      int x0_cond = (((x0_lid < 0) || ((x0_gid != 72) || (x0_tid < 1))) && ((x0_lid < 1) || (x0_gid != 72)));
      if (x0_cond)
      {
        int x0 = ((2 * x0_lid) + x0_tid);
        for (int c_lid = 0; c_lid < 2; c_lid += 1)
        {
          int c = ((32 * c_lid) + c_tid);
          float LX_T62 = agg[(c_lid + (x0_lid * 2))];
          LX_T62 = select((float)LX_T62, (float)0, (int)(LX_T62 == (float)-FLT_MAX));
          int gout_idx = ((c + (4672 * (x0_gid + x0))) + (64 * (x1_gid + x1_tid)));
          X_T62[gout_idx] = LX_T62;
        }
      }
    }
  }
}
