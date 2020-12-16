#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 9984 1 1
// lid: 256 1 1
// Original:
// X_T2139[n, d0, d1, c : _T3387, _T3388, _T3389, _T3390] = =(X_T2137[n, d0, d1, c])
// With Index Variables Made Integral:
// X_T2139[n, d0, d1, c : _T3387, _T3388, _T3389, _T3390] = =(X_T2137[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= d0 < 14, 0 <= d1 < 14, 0 <= d0 < 15, 0 <= d1 < 15, 0 <= c < 176, 0 <= c < 176, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= d0 < 14, 0 <= d1 < 14, 0 <= c < 176 }
// Defracted:
// X_T2139[n, d0, d1, c : _T3387, _T3388, _T3389, _T3390] = =(X_T2137[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range   X_T2139   X_T2137  
//        c       176         1         1  
//       d0        14      2640      2464  
//       d1        14       176       176  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 2464, 14 }
// Out stride: { 1, 2640 }
// Input 1 offset: 0
// Input 1 stride: { 1, 2464 }
// Tile size: { 64, 14 }
// Contraction output var shape: fp32(1, 15, 15, 176):(39600, 2640, 176, 1):154.688 KiB
// Computed true ops: 68992
// Computed work groups: 39
// Computed inner loops: 1
// Computed shared mem: 3640
// Computed out regs: 4096
// Computed mem read: 3584
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 9984, 1, 1
__kernel void kernel_c42_sdk_819(__global float* restrict  X_T2139, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[910];
  int d1_c_gid = (get_group_id(0) * 64);
  {
    {
      int d1_c_tid = (tid % 64);
      int d0_tid = ((tid / 64) % 4);
      for (int d0_lid = 0; d0_lid < 4; d0_lid += 1)
      {
        int d0_cond = ((d0_lid < 3) || (d0_tid < 2));
        if (d0_cond)
        {
          int d0 = ((4 * d0_lid) + d0_tid);
          int lidx = (d1_c_tid + (65 * d0));
          int gidx = ((d1_c_gid + d1_c_tid) + (2464 * d0));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)34495)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int d1_c_tid = (tid % 32);
    int d0_tid = ((tid / 32) % 8);
    for (int d1_c_lid = 0; d1_c_lid < 2; d1_c_lid += 1)
    {
      int d1_c = ((32 * d1_c_lid) + d1_c_tid);
      for (int d0_lid = 0; d0_lid < 2; d0_lid += 1)
      {
        int d0_cond = ((d0_lid < 1) || (d0_tid < 6));
        int d0 = select((int)0, (int)((8 * d0_lid) + d0_tid), (int)d0_cond);
        float val1 = in1_shared[(d1_c + (65 * d0))];
        int agg_idx = (d1_c_lid + (d0_lid * 2));
        agg[agg_idx] = select((float)agg[agg_idx], (float)val1, (int)d0_cond);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int d1_c_tid = (tid % 32);
  int d0_tid = ((tid / 32) % 8);
  for (int d1_c_lid = 0; d1_c_lid < 2; d1_c_lid += 1)
  {
    int d1_c_cond = ((d1_c_lid < 1) || (d1_c_gid != 2432));
    if (d1_c_cond)
    {
      int d1_c = ((32 * d1_c_lid) + d1_c_tid);
      for (int d0_lid = 0; d0_lid < 2; d0_lid += 1)
      {
        int d0_cond = ((d0_lid < 1) || (d0_tid < 6));
        if (d0_cond)
        {
          int d0 = ((8 * d0_lid) + d0_tid);
          float LX_T2139 = agg[(d1_c_lid + (d0_lid * 2))];
          int gout_idx = ((d1_c_gid + d1_c) + (2640 * d0));
          X_T2139[gout_idx] = LX_T2139;
        }
      }
    }
  }
}
