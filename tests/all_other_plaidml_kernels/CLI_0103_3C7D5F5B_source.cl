#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 55 1
// lid: 256 1 1
// Original:
// X_T250[n, d0, d1, c : _T337, _T338, _T339, _T340] = =(X_T249[n, -1 + d0, -1 + d1, c])
// With Index Variables Made Integral:
// X_T250[n, d0, d1, c : _T337, _T338, _T339, _T340] = =(X_T249[n, -1 + d0, -1 + d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= -1 + d0 < 83, 0 <= -1 + d1 < 83, 0 <= c < 84, 0 <= c < 84, 0 <= d0 < 85, 0 <= d1 < 85, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= -1 + d0 < 83, 0 <= -1 + d1 < 83, 0 <= c < 84 }
// Defracted:
// X_T250[n, d0, d1, c : _T337, _T338, _T339, _T340] = =(X_T249[n, -1 + d0, -1 + d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range    X_T250    X_T249  
//        c        84         1         1  
//       d0        83      7140      6972  
//       d1        83        84        84  
//      off                7224         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 6972, 83 }
// Out stride: { 1, 7140 }
// Input 1 offset: 0
// Input 1 stride: { 1, 6972 }
// Tile size: { 128, 4 }
// Contraction output var shape: fp32(1, 85, 85, 84):(606900, 7140, 84, 1):2370.7 KiB
// Computed true ops: 1157352
// Computed work groups: 1155
// Computed inner loops: 1
// Computed shared mem: 2064
// Computed out regs: 2048
// Computed mem read: 2048
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 55, 1
__kernel void kernel_c42_sdk_77(__global float* restrict  X_T250, __global const float* restrict  in1)
{
  X_T250 = (X_T250 + 7224);
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[516];
  int d1_c_gid = (get_group_id(1) * 128);
  int d0_gid = (get_group_id(0) * 4);
  {
    {
      int gbase = (d1_c_gid + (d0_gid * 6972));
      int d1_c_tid = (tid % 128);
      int d0_tid = ((tid / 128) % 2);
      for (int d0_lid = 0; d0_lid < 2; d0_lid += 1)
      {
        int d0 = ((2 * d0_lid) + d0_tid);
        int lidx = (d1_c_tid + (129 * d0));
        int gidx = ((gbase + d1_c_tid) + (6972 * d0));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)578675)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int d1_c_tid = (tid % 64);
    int d0_tid = ((tid / 64) % 4);
    for (int d1_c_lid = 0; d1_c_lid < 2; d1_c_lid += 1)
    {
      int d1_c = ((64 * d1_c_lid) + d1_c_tid);
      float val1 = in1_shared[(d1_c + (129 * d0_tid))];
      agg[d1_c_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int d1_c_tid = (tid % 64);
  int d0_tid = ((tid / 64) % 4);
  for (int d1_c_lid = 0; d1_c_lid < 2; d1_c_lid += 1)
  {
    int d1_c_cond = (((d1_c_lid < 0) || ((d1_c_gid != 6912) || (d1_c_tid < 60))) && ((d1_c_lid < 1) || (d1_c_gid != 6912)));
    if (d1_c_cond)
    {
      int d1_c = ((64 * d1_c_lid) + d1_c_tid);
      int d0_cond = ((d0_gid != 80) || (d0_tid < 3));
      if (d0_cond)
      {
        float LX_T250 = agg[d1_c_lid];
        int gout_idx = ((d1_c_gid + d1_c) + (7140 * (d0_gid + d0_tid)));
        X_T250[gout_idx] = LX_T250;
      }
    }
  }
}
