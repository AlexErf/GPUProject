#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 3 11
// lid: 256 1 1
// Original:
// X_T2978[n, o0, o1, c : _T4733, _T4734, _T4735, _T4736] = =(X_T100[])
// With Index Variables Made Integral:
// X_T2978[n, o0, o1, c : _T4733, _T4734, _T4735, _T4736] = =(X_T100[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 11, 0 <= o1 < 11, 0 <= c < 672, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 11, 0 <= o1 < 11, 0 <= c < 672 }
// Defracted:
// X_T2978[n, o0, o1, c : _T4733, _T4734, _T4735, _T4736] = =(X_T100[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range   X_T2978    X_T100  
//        c       672         1         0  
//       o0        11      7392         0  
//       o1        11       672         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 672, 11, 11 }
// Out stride: { 1, 7392, 672 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 256, 1, 4 }
// Contraction output var shape: fp32(1, 11, 11, 672):(81312, 7392, 672, 1):317.625 KiB
// Computed true ops: 162624
// Computed work groups: 99
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 4096
// Computed mem read: 128
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 3, 11
__kernel void kernel_c42_sdk_1150(__global float* restrict  X_T2978)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  int c_gid = (get_group_id(0) * 256);
  int o1_gid = (get_group_id(1) * 4);
  int o0_gid = get_group_id(2);
  {
    int c_tid = (tid % 64);
    int o1_tid = ((tid / 64) % 4);
    for (int c_lid = 0; c_lid < 4; c_lid += 1)
    {
      int c = ((64 * c_lid) + c_tid);
      float val1 = 1.0f;
      agg[c_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 64);
  int o1_tid = ((tid / 64) % 4);
  for (int c_lid = 0; c_lid < 4; c_lid += 1)
  {
    int c_cond = (((c_lid < 2) || ((c_gid != 512) || (c_tid < 32))) && ((c_lid < 3) || (c_gid != 512)));
    if (c_cond)
    {
      int c = ((64 * c_lid) + c_tid);
      int o1_cond = ((o1_gid != 8) || (o1_tid < 3));
      if (o1_cond)
      {
        float LX_T2978 = agg[c_lid];
        int gout_idx = (((c_gid + c) + (7392 * o0_gid)) + (672 * (o1_gid + o1_tid)));
        X_T2978[gout_idx] = LX_T2978;
      }
    }
  }
}
