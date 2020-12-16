#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 21 1
// lid: 256 1 1
// Original:
// X_T202[n, o0, o1, c : _T258, _T259, _T260, _T261] = =(X_T100[])
// With Index Variables Made Integral:
// X_T202[n, o0, o1, c : _T258, _T259, _T260, _T261] = =(X_T100[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= c < 42, 0 <= o0 < 83, 0 <= o1 < 83, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= c < 42, 0 <= o0 < 83, 0 <= o1 < 83 }
// Defracted:
// X_T202[n, o0, o1, c : _T258, _T259, _T260, _T261] = =(X_T100[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range    X_T202    X_T100  
//        c        42         1         0  
//       o0        83      3486         0  
//       o1        83        42         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 42, 83, 83 }
// Out stride: { 1, 3486, 42 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 42, 4, 4 }
// Contraction output var shape: fp32(1, 83, 83, 42):(289338, 3486, 42, 1):1130.23 KiB
// Computed true ops: 578676
// Computed work groups: 441
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 4096
// Computed mem read: 128
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 21, 1
__kernel void kernel_c42_sdk_58(__global float* restrict  X_T202)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  int o1_gid = (get_group_id(0) * 4);
  int o0_gid = (get_group_id(1) * 4);
  {
    int c_tid = (tid % 32);
    int o1_tid = ((tid / 32) % 4);
    int o0_tid = ((tid / 128) % 2);
    for (int o0_lid = 0; o0_lid < 2; o0_lid += 1)
    {
      int o0 = ((2 * o0_lid) + o0_tid);
      for (int c_lid = 0; c_lid < 2; c_lid += 1)
      {
        int c_cond = ((c_lid < 1) || (c_tid < 10));
        int c = select((int)0, (int)((32 * c_lid) + c_tid), (int)c_cond);
        float val1 = 1.0f;
        int agg_idx = (c_lid + (o0_lid * 2));
        agg[agg_idx] = select((float)agg[agg_idx], (float)val1, (int)c_cond);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o1_tid = ((tid / 32) % 4);
  int o0_tid = ((tid / 128) % 2);
  int o1_cond = ((o1_gid != 80) || (o1_tid < 3));
  if (o1_cond)
  {
    for (int o0_lid = 0; o0_lid < 2; o0_lid += 1)
    {
      int o0_cond = ((o0_lid < 1) || ((o0_gid != 80) || (o0_tid < 1)));
      if (o0_cond)
      {
        int o0 = ((2 * o0_lid) + o0_tid);
        for (int c_lid = 0; c_lid < 2; c_lid += 1)
        {
          int c_cond = ((c_lid < 1) || (c_tid < 10));
          if (c_cond)
          {
            int c = ((32 * c_lid) + c_tid);
            float LX_T202 = agg[(c_lid + (o0_lid * 2))];
            int gout_idx = ((c + (3486 * (o0_gid + o0))) + (42 * (o1_gid + o1_tid)));
            X_T202[gout_idx] = LX_T202;
          }
        }
      }
    }
  }
}
