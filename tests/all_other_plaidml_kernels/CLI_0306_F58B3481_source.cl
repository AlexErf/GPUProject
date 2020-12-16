#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Original:
// X_T574[n, o0, o1, c : _T749, _T750, _T751, _T752] = =(X_T253[])
// With Index Variables Made Integral:
// X_T574[n, o0, o1, c : _T749, _T750, _T751, _T752] = =(X_T253[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 28, 0 <= o1 < 28, 0 <= c < 256, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 28, 0 <= o1 < 28, 0 <= c < 256 }
// Defracted:
// X_T574[n, o0, o1, c : _T749, _T750, _T751, _T752] = =(X_T253[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range    X_T574    X_T253  
//        c       256         1         0  
//       o0        28      7168         0  
//       o1        28       256         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 256, 28, 28 }
// Out stride: { 1, 7168, 256 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 256, 4, 4 }
// Contraction output var shape: fp32(1, 28, 28, 256):(200704, 7168, 256, 1):784 KiB
// Computed true ops: 401408
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 16384
// Computed mem read: 128
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_178(__global float* restrict  X_T574)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  int o1_gid = (get_group_id(0) * 4);
  int o0_gid = (get_group_id(1) * 4);
  {
    int c_tid = (tid % 32);
    int o1_tid = ((tid / 32) % 4);
    int o0_tid = ((tid / 128) % 2);
    for (int c_lid = 0; c_lid < 8; c_lid += 1)
    {
      int c = ((32 * c_lid) + c_tid);
      for (int o0_lid = 0; o0_lid < 2; o0_lid += 1)
      {
        int o0 = ((2 * o0_lid) + o0_tid);
        float val1 = 1.0f;
        int agg_idx = (c_lid + (o0_lid * 8));
        agg[agg_idx] = val1;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o1_tid = ((tid / 32) % 4);
  int o0_tid = ((tid / 128) % 2);
  for (int c_lid = 0; c_lid < 8; c_lid += 1)
  {
    int c = ((32 * c_lid) + c_tid);
    for (int o0_lid = 0; o0_lid < 2; o0_lid += 1)
    {
      int o0 = ((2 * o0_lid) + o0_tid);
      float LX_T574 = agg[(c_lid + (o0_lid * 8))];
      int gout_idx = ((c + (7168 * (o0_gid + o0))) + (256 * (o1_gid + o1_tid)));
      X_T574[gout_idx] = LX_T574;
    }
  }
}
