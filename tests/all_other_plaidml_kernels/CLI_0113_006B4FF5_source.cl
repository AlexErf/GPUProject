#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 9 9
// lid: 256 1 1
// Original:
// X_T84[n, x0, x1, c : _T74, _T75, _T76, _T77] = >(X_T83[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T84[n, x0, x1, c : _T74, _T75, _T76, _T77] = >(X_T83[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 35, 0 <= x1 < 35, 0 <= k0 + 2*x0 < 71, 0 <= k1 + 2*x1 < 71, 0 <= c < 192, 0 <= c < 192, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 35, 0 <= x1 < 35, 0 <= k0 + 2*x0 < 71, 0 <= k1 + 2*x1 < 71, 0 <= c < 192 }
// Defracted:
// X_T84[n, x0, x1, c : _T74, _T75, _T76, _T77] = >(X_T83[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T84     X_T83  
//        c       192         1         1  
//       k0         3         0     13632  
//       k1         3         0       192  
//       x0        35      6720     27264  
//       x1        35       192       384  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 192, 3, 3, 35, 35 }
// Out stride: { 1, 0, 0, 6720, 192 }
// Input 1 offset: 0
// Input 1 stride: { 1, 13632, 192, 27264, 384 }
// Tile size: { 64, 3, 3, 4, 4 }
// Contraction output var shape: fp32(1, 35, 35, 192):(235200, 6720, 192, 1):918.75 KiB
// Computed true ops: 4233600
// Computed work groups: 243
// Computed inner loops: 1
// Computed shared mem: 20736
// Computed out regs: 4096
// Computed mem read: 20736
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 9, 9
__kernel void kernel_c56_sdk_16(__global float* restrict  X_T84, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, };
  __local float in1_shared[5184];
  int c_gid = (get_group_id(0) * 64);
  int x1_gid = (get_group_id(1) * 4);
  int x0_gid = (get_group_id(2) * 4);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((((c_gid + (k1_gid * 192)) + (x1_gid * 384)) + (k0_gid * 13632)) + (x0_gid * 27264));
        int c_tid = (tid % 64);
        int k1_x1_tid = ((tid / 64) % 4);
        for (int k1_x1_lid = 0; k1_x1_lid < 3; k1_x1_lid += 1)
        {
          int k1_x1_cond = ((k1_x1_lid < 2) || (k1_x1_tid < 1));
          if (k1_x1_cond)
          {
            int k1_x1 = ((4 * k1_x1_lid) + k1_x1_tid);
            for (int k0_x0_lid = 0; k0_x0_lid < 9; k0_x0_lid += 1)
            {
              int lidx = (((81 * c_tid) + k1_x1) + (9 * k0_x0_lid));
              int gidx = (((gbase + c_tid) + (192 * k1_x1)) + (13632 * k0_x0_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)967871)];
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
              float val1 = in1_shared[(((((81 * c) + k1_lid) + (2 * x1_tid)) + (9 * k0_lid)) + (18 * x0))];
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
  int x1_cond = ((x1_gid != 32) || (x1_tid < 3));
  if (x1_cond)
  {
    for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
    {
      int x0_cond = ((x0_lid < 1) || ((x0_gid != 32) || (x0_tid < 1)));
      if (x0_cond)
      {
        int x0 = ((2 * x0_lid) + x0_tid);
        for (int c_lid = 0; c_lid < 2; c_lid += 1)
        {
          int c = ((32 * c_lid) + c_tid);
          float LX_T84 = agg[(c_lid + (x0_lid * 2))];
          LX_T84 = select((float)LX_T84, (float)0, (int)(LX_T84 == (float)-FLT_MAX));
          int gout_idx = (((c_gid + c) + (6720 * (x0_gid + x0))) + (192 * (x1_gid + x1_tid)));
          X_T84[gout_idx] = LX_T84;
        }
      }
    }
  }
}
