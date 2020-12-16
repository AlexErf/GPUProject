#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2304 9 1
// lid: 256 1 1
// Original:
// X_T381[n, x0, x1, c : _T523, _T524, _T525, _T526] = >(X_T337[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T381[n, x0, x1, c : _T523, _T524, _T525, _T526] = >(X_T337[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 17, 0 <= x1 < 17, 0 <= k0 + 2*x0 < 35, 0 <= k1 + 2*x1 < 35, 0 <= c < 288, 0 <= c < 288, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 17, 0 <= x1 < 17, 0 <= k0 + 2*x0 < 35, 0 <= k1 + 2*x1 < 35, 0 <= c < 288 }
// Defracted:
// X_T381[n, x0, x1, c : _T523, _T524, _T525, _T526] = >(X_T337[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T381    X_T337  
//        c       288         1         1  
//       k0         3         0     10080  
//       k1         3         0       288  
//       x0        17      4896     20160  
//       x1        17       288       576  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 288, 3, 3, 17, 17 }
// Out stride: { 1, 0, 0, 4896, 288 }
// Input 1 offset: 0
// Input 1 stride: { 1, 10080, 288, 20160, 576 }
// Tile size: { 32, 3, 3, 2, 17 }
// Contraction output var shape: fp32(1, 17, 17, 288):(83232, 4896, 288, 1):325.125 KiB
// Computed true ops: 1498176
// Computed work groups: 81
// Computed inner loops: 1
// Computed shared mem: 22400
// Computed out regs: 5120
// Computed mem read: 22400
// Computed mem write: 4352
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2304, 9, 1
__kernel void kernel_c56_sdk_125(__global float* restrict  X_T381, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[5] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, };
  __local float in1_shared[5600];
  int c_gid = (get_group_id(0) * 32);
  int x0_gid = (get_group_id(1) * 2);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = (((c_gid + (k1_gid * 288)) + (k0_gid * 10080)) + (x0_gid * 20160));
        int c_tid = (tid % 32);
        int k1_x1_k0_x0_tid = ((tid / 32) % 8);
        for (int k1_x1_k0_x0_lid = 0; k1_x1_k0_x0_lid < 22; k1_x1_k0_x0_lid += 1)
        {
          int k1_x1_k0_x0_cond = ((k1_x1_k0_x0_lid < 21) || (k1_x1_k0_x0_tid < 7));
          if (k1_x1_k0_x0_cond)
          {
            int k1_x1_k0_x0 = ((8 * k1_x1_k0_x0_lid) + k1_x1_k0_x0_tid);
            int lidx = ((175 * c_tid) + k1_x1_k0_x0);
            int gidx = ((gbase + c_tid) + (288 * k1_x1_k0_x0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)352799)];
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
          for (int x1_lid = 0; x1_lid < 5; x1_lid += 1)
          {
            int x1_cond = ((x1_lid < 4) || (x1_tid < 1));
            int x1 = select((int)0, (int)((4 * x1_lid) + x1_tid), (int)x1_cond);
            float val1 = in1_shared[(((((175 * c_tid) + k1_lid) + (2 * x1)) + (35 * k0_lid)) + (70 * x0_tid))];
            float agg_rhs = select((float)agg[x1_lid], (float)val1, (int)(val1 > agg[x1_lid]));
            agg[x1_lid] = select((float)agg[x1_lid], (float)agg_rhs, (int)x1_cond);
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 4);
  int x0_tid = ((tid / 128) % 2);
  int x0_cond = ((x0_gid != 16) || (x0_tid < 1));
  if (x0_cond)
  {
    for (int x1_lid = 0; x1_lid < 5; x1_lid += 1)
    {
      int x1_cond = ((x1_lid < 4) || (x1_tid < 1));
      if (x1_cond)
      {
        int x1 = ((4 * x1_lid) + x1_tid);
        float LX_T381 = agg[x1_lid];
        LX_T381 = select((float)LX_T381, (float)0, (int)(LX_T381 == (float)-FLT_MAX));
        int gout_idx = (((c_gid + c_tid) + (4896 * (x0_gid + x0_tid))) + (288 * x1));
        X_T381[gout_idx] = LX_T381;
      }
    }
  }
}
