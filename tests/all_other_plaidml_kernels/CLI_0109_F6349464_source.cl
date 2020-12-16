#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 10752 1 1
// lid: 256 1 1
// Original:
// X_T305[n, d0, d1, c : _T333, _T334, _T335, _T336] = =(X_T303[n, d0, d1, c])
// With Index Variables Made Integral:
// X_T305[n, d0, d1, c : _T333, _T334, _T335, _T336] = =(X_T303[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= d0 < 28, 0 <= d1 < 28, 0 <= d0 < 29, 0 <= d1 < 29, 0 <= c < 192, 0 <= c < 192, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= d0 < 28, 0 <= d1 < 28, 0 <= c < 192 }
// Defracted:
// X_T305[n, d0, d1, c : _T333, _T334, _T335, _T336] = =(X_T303[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range    X_T305    X_T303  
//        c       192         1         1  
//       d0        28      5568      5376  
//       d1        28       192       192  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 5376, 28 }
// Out stride: { 1, 5568 }
// Input 1 offset: 0
// Input 1 stride: { 1, 5376 }
// Tile size: { 128, 28 }
// Contraction output var shape: fp32(1, 29, 29, 192):(161472, 5568, 192, 1):630.75 KiB
// Computed true ops: 301056
// Computed work groups: 42
// Computed inner loops: 1
// Computed shared mem: 14448
// Computed out regs: 16384
// Computed mem read: 14336
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 10752, 1, 1
__kernel void kernel_c43_sdk_76(__global float* restrict  X_T305, __global const float* restrict  in1)
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
        int gidx = ((d1_c_gid + d1_c_tid) + (5376 * d0));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)150527)];
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
        float LX_T305 = agg[(d1_c_lid + (d0_lid * 4))];
        int gout_idx = ((d1_c_gid + d1_c) + (5568 * d0));
        X_T305[gout_idx] = LX_T305;
      }
    }
  }
}
