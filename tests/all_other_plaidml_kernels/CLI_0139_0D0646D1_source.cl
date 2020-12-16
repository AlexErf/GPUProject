#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 9 9
// lid: 256 1 1
// Original:
// X_T930[n, x0, x1, c : _T1280, _T1281, _T1282, _T1283] = >(X_T886[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T930[n, x0, x1, c : _T1280, _T1281, _T1282, _T1283] = >(X_T886[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 17, 0 <= x1 < 17, 0 <= k0 + 2*x0 < 35, 0 <= k1 + 2*x1 < 35, 0 <= c < 320, 0 <= c < 320, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 17, 0 <= x1 < 17, 0 <= k0 + 2*x0 < 35, 0 <= k1 + 2*x1 < 35, 0 <= c < 320 }
// Defracted:
// X_T930[n, x0, x1, c : _T1280, _T1281, _T1282, _T1283] = >(X_T886[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T930    X_T886  
//        c       320         1         1  
//       k0         3         0     11200  
//       k1         3         0       320  
//       x0        17      5440     22400  
//       x1        17       320       640  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 320, 3, 3, 17, 17 }
// Out stride: { 1, 0, 0, 5440, 320 }
// Input 1 offset: 0
// Input 1 stride: { 1, 11200, 320, 22400, 640 }
// Tile size: { 64, 3, 3, 2, 2 }
// Contraction output var shape: fp32(1, 17, 17, 320):(92480, 5440, 320, 1):361.25 KiB
// Computed true ops: 1664640
// Computed work groups: 405
// Computed inner loops: 1
// Computed shared mem: 6400
// Computed out regs: 1024
// Computed mem read: 6400
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1280, 9, 9
__kernel void kernel_c51_sdk_303(__global float* restrict  X_T930, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[1] = {-FLT_MAX, };
  __local float in1_shared[1600];
  int c_gid = (get_group_id(0) * 64);
  int x1_gid = (get_group_id(1) * 2);
  int x0_gid = (get_group_id(2) * 2);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((((c_gid + (k1_gid * 320)) + (x1_gid * 640)) + (k0_gid * 11200)) + (x0_gid * 22400));
        int c_tid = (tid % 64);
        int k1_x1_tid = ((tid / 64) % 4);
        for (int k1_x1_lid = 0; k1_x1_lid < 2; k1_x1_lid += 1)
        {
          int k1_x1_cond = ((k1_x1_lid < 1) || (k1_x1_tid < 1));
          if (k1_x1_cond)
          {
            int k1_x1 = ((4 * k1_x1_lid) + k1_x1_tid);
            for (int k0_x0_lid = 0; k0_x0_lid < 5; k0_x0_lid += 1)
            {
              int lidx = (((25 * c_tid) + k1_x1) + (5 * k0_x0_lid));
              int gidx = (((gbase + c_tid) + (320 * k1_x1)) + (11200 * k0_x0_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)391999)];
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
      {
        for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
        {
          int c_tid = (tid % 64);
          int x1_tid = ((tid / 64) % 2);
          int x0_tid = ((tid / 128) % 2);
          float val1 = in1_shared[(((((25 * c_tid) + k1_lid) + (2 * x1_tid)) + (5 * k0_lid)) + (10 * x0_tid))];
          float agg_rhs = select((float)agg[0], (float)val1, (int)(val1 > agg[0]));
          agg[0] = agg_rhs;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 64);
  int x1_tid = ((tid / 64) % 2);
  int x0_tid = ((tid / 128) % 2);
  int x1_cond = ((x1_gid != 16) || (x1_tid < 1));
  if (x1_cond)
  {
    int x0_cond = ((x0_gid != 16) || (x0_tid < 1));
    if (x0_cond)
    {
      float LX_T930 = agg[0];
      LX_T930 = select((float)LX_T930, (float)0, (int)(LX_T930 == (float)-FLT_MAX));
      int gout_idx = (((c_gid + c_tid) + (5440 * (x0_gid + x0_tid))) + (320 * (x1_gid + x1_tid)));
      X_T930[gout_idx] = LX_T930;
    }
  }
}
