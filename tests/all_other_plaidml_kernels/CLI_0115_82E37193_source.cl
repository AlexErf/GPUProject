#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2304 9 1
// lid: 256 1 1
// Original:
// X_T161[n, o0, o1, c : _T179, _T180, _T181, _T182] = =(X_T160[])
// With Index Variables Made Integral:
// X_T161[n, o0, o1, c : _T179, _T180, _T181, _T182] = =(X_T160[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 35, 0 <= o1 < 35, 0 <= c < 192, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 35, 0 <= o1 < 35, 0 <= c < 192 }
// Defracted:
// X_T161[n, o0, o1, c : _T179, _T180, _T181, _T182] = =(X_T160[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range    X_T161    X_T160  
//        c       192         1         0  
//       o0        35      6720         0  
//       o1        35       192         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 192, 35, 35 }
// Out stride: { 1, 6720, 192 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 192, 4, 4 }
// Contraction output var shape: fp32(1, 35, 35, 192):(235200, 6720, 192, 1):918.75 KiB
// Computed true ops: 470400
// Computed work groups: 81
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 12288
// Computed mem read: 128
// Computed mem write: 12288
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2304, 9, 1
__kernel void kernel_c51_sdk_41(__global float* restrict  X_T161)
{
  int tid = get_local_id(0);
  float agg[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  int o1_gid = (get_group_id(0) * 4);
  int o0_gid = (get_group_id(1) * 4);
  {
    int c_tid = (tid % 32);
    int o1_tid = ((tid / 32) % 4);
    int o0_tid = ((tid / 128) % 2);
    for (int o0_lid = 0; o0_lid < 2; o0_lid += 1)
    {
      int o0 = ((2 * o0_lid) + o0_tid);
      for (int c_lid = 0; c_lid < 6; c_lid += 1)
      {
        int c = ((32 * c_lid) + c_tid);
        float val1 = 1.0f;
        int agg_idx = (c_lid + (o0_lid * 6));
        agg[agg_idx] = val1;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o1_tid = ((tid / 32) % 4);
  int o0_tid = ((tid / 128) % 2);
  int o1_cond = ((o1_gid != 32) || (o1_tid < 3));
  if (o1_cond)
  {
    for (int o0_lid = 0; o0_lid < 2; o0_lid += 1)
    {
      int o0_cond = ((o0_lid < 1) || ((o0_gid != 32) || (o0_tid < 1)));
      if (o0_cond)
      {
        int o0 = ((2 * o0_lid) + o0_tid);
        for (int c_lid = 0; c_lid < 6; c_lid += 1)
        {
          int c = ((32 * c_lid) + c_tid);
          float LX_T161 = agg[(c_lid + (o0_lid * 6))];
          int gout_idx = ((c + (6720 * (o0_gid + o0))) + (192 * (o1_gid + o1_tid)));
          X_T161[gout_idx] = LX_T161;
        }
      }
    }
  }
}
