#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 28672 1 1
// lid: 256 1 1
// Original:
// X_T171[n, d0, d1, c : _T158, _T159, _T160, _T161] = =(X_T169[n, d0, d1, c])
// With Index Variables Made Integral:
// X_T171[n, d0, d1, c : _T158, _T159, _T160, _T161] = =(X_T169[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= d0 < 56, 0 <= d1 < 56, 0 <= d0 < 57, 0 <= d1 < 57, 0 <= c < 128, 0 <= c < 128, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= d0 < 56, 0 <= d1 < 56, 0 <= c < 128 }
// Defracted:
// X_T171[n, d0, d1, c : _T158, _T159, _T160, _T161] = =(X_T169[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range    X_T171    X_T169  
//        c       128         1         1  
//       d0        56      7296      7168  
//       d1        56       128       128  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 7168, 56 }
// Out stride: { 1, 7296 }
// Input 1 offset: 0
// Input 1 stride: { 1, 7168 }
// Tile size: { 64, 56 }
// Contraction output var shape: fp32(1, 57, 57, 128):(415872, 7296, 128, 1):1624.5 KiB
// Computed true ops: 802816
// Computed work groups: 112
// Computed inner loops: 1
// Computed shared mem: 14560
// Computed out regs: 14336
// Computed mem read: 14336
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 28672, 1, 1
__kernel void kernel_c25_sdk_40(__global float* restrict  X_T171, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3640];
  int d1_c_gid = (get_group_id(0) * 64);
  {
    {
      int d1_c_tid = (tid % 64);
      int d0_tid = ((tid / 64) % 4);
      for (int d0_lid = 0; d0_lid < 14; d0_lid += 1)
      {
        int d0 = ((4 * d0_lid) + d0_tid);
        int lidx = (d1_c_tid + (65 * d0));
        int gidx = ((d1_c_gid + d1_c_tid) + (7168 * d0));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)401407)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int d1_c_tid = (tid % 32);
    int d0_tid = ((tid / 32) % 8);
    for (int d1_c_lid = 0; d1_c_lid < 2; d1_c_lid += 1)
    {
      int d1_c = ((32 * d1_c_lid) + d1_c_tid);
      for (int d0_lid = 0; d0_lid < 7; d0_lid += 1)
      {
        int d0 = ((8 * d0_lid) + d0_tid);
        float val1 = in1_shared[(d1_c + (65 * d0))];
        int agg_idx = (d1_c_lid + (d0_lid * 2));
        agg[agg_idx] = val1;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int d1_c_tid = (tid % 32);
  int d0_tid = ((tid / 32) % 8);
  for (int d1_c_lid = 0; d1_c_lid < 2; d1_c_lid += 1)
  {
    int d1_c = ((32 * d1_c_lid) + d1_c_tid);
    for (int d0_lid = 0; d0_lid < 7; d0_lid += 1)
    {
      int d0 = ((8 * d0_lid) + d0_tid);
      float LX_T171 = agg[(d1_c_lid + (d0_lid * 2))];
      int gout_idx = ((d1_c_gid + d1_c) + (7296 * d0));
      X_T171[gout_idx] = LX_T171;
    }
  }
}
