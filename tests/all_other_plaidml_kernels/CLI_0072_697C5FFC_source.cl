#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 28416 1 1
// lid: 256 1 1
// Original:
// X_T69[n, d0, d1, c : _T50, _T51, _T52, _T53] = =(X_T45[n, -3 + d0, -3 + d1, c])
// With Index Variables Made Integral:
// X_T69[n, d0, d1, c : _T50, _T51, _T52, _T53] = =(X_T45[n, -3 + d0, -3 + d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= c < 32, 0 <= c < 32, 0 <= -3 + d0 < 111, 0 <= -3 + d1 < 111, 0 <= d0 < 117, 0 <= d1 < 117, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= c < 32, 0 <= -3 + d0 < 111, 0 <= -3 + d1 < 111 }
// Defracted:
// X_T69[n, d0, d1, c : _T50, _T51, _T52, _T53] = =(X_T45[n, -3 + d0, -3 + d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range     X_T69     X_T45  
//        c        32         1         1  
//       d0       111      3744      3552  
//       d1       111        32        32  
//      off               11328         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 3552, 111 }
// Out stride: { 1, 3744 }
// Input 1 offset: 0
// Input 1 stride: { 1, 3552 }
// Tile size: { 32, 111 }
// Contraction output var shape: fp32(1, 117, 117, 32):(438048, 3744, 32, 1):1711.12 KiB
// Computed true ops: 788544
// Computed work groups: 111
// Computed inner loops: 1
// Computed shared mem: 14208
// Computed out regs: 14336
// Computed mem read: 14208
// Computed mem write: 14208
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 28416, 1, 1
__kernel void kernel_c42_sdk_10(__global float* restrict  X_T69, __global const float* restrict  in1)
{
  X_T69 = (X_T69 + 11328);
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3552];
  int d1_c_gid = (get_group_id(0) * 32);
  {
    {
      int d1_c_tid = (tid % 32);
      int d0_tid = ((tid / 32) % 8);
      for (int d0_lid = 0; d0_lid < 14; d0_lid += 1)
      {
        int d0_cond = ((d0_lid < 13) || (d0_tid < 7));
        if (d0_cond)
        {
          int d0 = ((8 * d0_lid) + d0_tid);
          int lidx = ((111 * d1_c_tid) + d0);
          int gidx = ((d1_c_gid + d1_c_tid) + (3552 * d0));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)394271)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int d1_c_tid = (tid % 32);
    int d0_tid = ((tid / 32) % 8);
    for (int d0_lid = 0; d0_lid < 14; d0_lid += 1)
    {
      int d0_cond = ((d0_lid < 13) || (d0_tid < 7));
      int d0 = select((int)0, (int)((8 * d0_lid) + d0_tid), (int)d0_cond);
      float val1 = in1_shared[((111 * d1_c_tid) + d0)];
      agg[d0_lid] = select((float)agg[d0_lid], (float)val1, (int)d0_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int d1_c_tid = (tid % 32);
  int d0_tid = ((tid / 32) % 8);
  for (int d0_lid = 0; d0_lid < 14; d0_lid += 1)
  {
    int d0_cond = ((d0_lid < 13) || (d0_tid < 7));
    if (d0_cond)
    {
      int d0 = ((8 * d0_lid) + d0_tid);
      float LX_T69 = agg[d0_lid];
      int gout_idx = ((d1_c_gid + d1_c_tid) + (3744 * d0));
      X_T69[gout_idx] = LX_T69;
    }
  }
}
