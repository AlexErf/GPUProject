#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 16128 1 1
// lid: 256 1 1
// Original:
// X_T569[n, d0, d1, c : _T669, _T670, _T671, _T672] = =(X_T567[n, d0, d1, c])
// With Index Variables Made Integral:
// X_T569[n, d0, d1, c : _T669, _T670, _T671, _T672] = =(X_T567[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= d0 < 14, 0 <= d1 < 14, 0 <= d0 < 15, 0 <= d1 < 15, 0 <= c < 576, 0 <= c < 576, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= d0 < 14, 0 <= d1 < 14, 0 <= c < 576 }
// Defracted:
// X_T569[n, d0, d1, c : _T669, _T670, _T671, _T672] = =(X_T567[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range    X_T569    X_T567  
//        c       576         1         1  
//       d0        14      8640      8064  
//       d1        14       576       576  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 8064, 14 }
// Out stride: { 1, 8640 }
// Input 1 offset: 0
// Input 1 stride: { 1, 8064 }
// Tile size: { 128, 14 }
// Contraction output var shape: fp32(1, 15, 15, 576):(129600, 8640, 576, 1):506.25 KiB
// Computed true ops: 225792
// Computed work groups: 63
// Computed inner loops: 1
// Computed shared mem: 7224
// Computed out regs: 8192
// Computed mem read: 7168
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 16128, 1, 1
__kernel void kernel_c43_sdk_152(__global float* restrict  X_T569, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[8] = {0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[1806];
  int d1_c_gid = (get_group_id(0) * 128);
  {
    {
      int d1_c_tid = (tid % 128);
      int d0_tid = ((tid / 128) % 2);
      for (int d0_lid = 0; d0_lid < 7; d0_lid += 1)
      {
        int d0 = ((2 * d0_lid) + d0_tid);
        int lidx = (d1_c_tid + (129 * d0));
        int gidx = ((d1_c_gid + d1_c_tid) + (8064 * d0));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)112895)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int d1_c_tid = (tid % 32);
    int d0_tid = ((tid / 32) % 8);
    for (int d0_lid = 0; d0_lid < 2; d0_lid += 1)
    {
      int d0_cond = ((d0_lid < 1) || (d0_tid < 6));
      int d0 = select((int)0, (int)((8 * d0_lid) + d0_tid), (int)d0_cond);
      for (int d1_c_lid = 0; d1_c_lid < 4; d1_c_lid += 1)
      {
        int d1_c = ((32 * d1_c_lid) + d1_c_tid);
        float val1 = in1_shared[(d1_c + (129 * d0))];
        int agg_idx = (d1_c_lid + (d0_lid * 4));
        agg[agg_idx] = select((float)agg[agg_idx], (float)val1, (int)d0_cond);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int d1_c_tid = (tid % 32);
  int d0_tid = ((tid / 32) % 8);
  for (int d0_lid = 0; d0_lid < 2; d0_lid += 1)
  {
    int d0_cond = ((d0_lid < 1) || (d0_tid < 6));
    if (d0_cond)
    {
      int d0 = ((8 * d0_lid) + d0_tid);
      for (int d1_c_lid = 0; d1_c_lid < 4; d1_c_lid += 1)
      {
        int d1_c = ((32 * d1_c_lid) + d1_c_tid);
        float LX_T569 = agg[(d1_c_lid + (d0_lid * 4))];
        int gout_idx = ((d1_c_gid + d1_c) + (8640 * d0));
        X_T569[gout_idx] = LX_T569;
      }
    }
  }
}
