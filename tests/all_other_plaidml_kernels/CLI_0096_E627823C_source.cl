#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 63 1
// lid: 256 1 1
// Original:
// X_T187[n, d0, d1, c : _T187, _T188, _T189, _T190] = =(X_T185[n, d0, d1, c])
// With Index Variables Made Integral:
// X_T187[n, d0, d1, c : _T187, _T188, _T189, _T190] = =(X_T185[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= d0 < 56, 0 <= d1 < 56, 0 <= d0 < 57, 0 <= d1 < 57, 0 <= c < 144, 0 <= c < 144, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= d0 < 56, 0 <= d1 < 56, 0 <= c < 144 }
// Defracted:
// X_T187[n, d0, d1, c : _T187, _T188, _T189, _T190] = =(X_T185[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range    X_T187    X_T185  
//        c       144         1         1  
//       d0        56      8208      8064  
//       d1        56       144       144  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 8064, 56 }
// Out stride: { 1, 8208 }
// Input 1 offset: 0
// Input 1 stride: { 1, 8064 }
// Tile size: { 128, 8 }
// Contraction output var shape: fp32(1, 57, 57, 144):(467856, 8208, 144, 1):1827.56 KiB
// Computed true ops: 903168
// Computed work groups: 441
// Computed inner loops: 1
// Computed shared mem: 4128
// Computed out regs: 4096
// Computed mem read: 4096
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 63, 1
__kernel void kernel_c43_sdk_43(__global float* restrict  X_T187, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[1032];
  int d1_c_gid = (get_group_id(1) * 128);
  int d0_gid = (get_group_id(0) * 8);
  {
    {
      int gbase = (d1_c_gid + (d0_gid * 8064));
      int d1_c_tid = (tid % 128);
      int d0_tid = ((tid / 128) % 2);
      for (int d0_lid = 0; d0_lid < 4; d0_lid += 1)
      {
        int d0 = ((2 * d0_lid) + d0_tid);
        int lidx = (d1_c_tid + (129 * d0));
        int gidx = ((gbase + d1_c_tid) + (8064 * d0));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)451583)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int d1_c_tid = (tid % 32);
    int d0_tid = ((tid / 32) % 8);
    for (int d1_c_lid = 0; d1_c_lid < 4; d1_c_lid += 1)
    {
      int d1_c = ((32 * d1_c_lid) + d1_c_tid);
      float val1 = in1_shared[(d1_c + (129 * d0_tid))];
      agg[d1_c_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int d1_c_tid = (tid % 32);
  int d0_tid = ((tid / 32) % 8);
  for (int d1_c_lid = 0; d1_c_lid < 4; d1_c_lid += 1)
  {
    int d1_c = ((32 * d1_c_lid) + d1_c_tid);
    float LX_T187 = agg[d1_c_lid];
    int gout_idx = ((d1_c_gid + d1_c) + (8208 * (d0_gid + d0_tid)));
    X_T187[gout_idx] = LX_T187;
  }
}
