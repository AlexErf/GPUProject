#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 14336 1 1
// lid: 256 1 1
// Original:
// X_T227[n, d0, d1, c : _T230, _T231, _T232, _T233] = =(X_T225[n, d0, d1, c])
// With Index Variables Made Integral:
// X_T227[n, d0, d1, c : _T230, _T231, _T232, _T233] = =(X_T225[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= d0 < 28, 0 <= d1 < 28, 0 <= d0 < 29, 0 <= d1 < 29, 0 <= c < 256, 0 <= c < 256, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= d0 < 28, 0 <= d1 < 28, 0 <= c < 256 }
// Defracted:
// X_T227[n, d0, d1, c : _T230, _T231, _T232, _T233] = =(X_T225[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range    X_T227    X_T225  
//        c       256         1         1  
//       d0        28      7424      7168  
//       d1        28       256       256  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 7168, 28 }
// Out stride: { 1, 7424 }
// Input 1 offset: 0
// Input 1 stride: { 1, 7168 }
// Tile size: { 128, 28 }
// Contraction output var shape: fp32(1, 29, 29, 256):(215296, 7424, 256, 1):841 KiB
// Computed true ops: 401408
// Computed work groups: 56
// Computed inner loops: 1
// Computed shared mem: 14448
// Computed out regs: 16384
// Computed mem read: 14336
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 14336, 1, 1
__kernel void kernel_c25_sdk_55(__global float* restrict  X_T227, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3612];
  int d1_c_gid = (get_group_id(0) * 128);
  {
    {
      int d1_c_tid = (tid % 128);
      int d0_tid = ((tid / 128) % 2);
      for (int d0_lid = 0; d0_lid < 14; d0_lid += 1)
      {
        int d0 = ((2 * d0_lid) + d0_tid);
        int lidx = (d1_c_tid + (129 * d0));
        int gidx = ((d1_c_gid + d1_c_tid) + (7168 * d0));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)200703)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int d1_c_tid = (tid % 32);
    int d0_tid = ((tid / 32) % 8);
    for (int d0_lid = 0; d0_lid < 4; d0_lid += 1)
    {
      int d0_cond = ((d0_lid < 3) || (d0_tid < 4));
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
  for (int d0_lid = 0; d0_lid < 4; d0_lid += 1)
  {
    int d0_cond = ((d0_lid < 3) || (d0_tid < 4));
    if (d0_cond)
    {
      int d0 = ((8 * d0_lid) + d0_tid);
      for (int d1_c_lid = 0; d1_c_lid < 4; d1_c_lid += 1)
      {
        int d1_c = ((32 * d1_c_lid) + d1_c_tid);
        float LX_T227 = agg[(d1_c_lid + (d0_lid * 4))];
        int gout_idx = ((d1_c_gid + d1_c) + (7424 * d0));
        X_T227[gout_idx] = LX_T227;
      }
    }
  }
}
