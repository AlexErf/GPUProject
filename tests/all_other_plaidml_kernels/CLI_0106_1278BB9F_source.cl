#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 83 1
// lid: 256 1 1
// Original:
// X_T260[n, o0, o1, c : _T356, _T357, _T358, _T359] = =(X_T100[])
// With Index Variables Made Integral:
// X_T260[n, o0, o1, c : _T356, _T357, _T358, _T359] = =(X_T100[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= c < 96, 0 <= o0 < 165, 0 <= o1 < 165, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= c < 96, 0 <= o0 < 165, 0 <= o1 < 165 }
// Defracted:
// X_T260[n, o0, o1, c : _T356, _T357, _T358, _T359] = =(X_T100[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range    X_T260    X_T100  
//        c        96         1         0  
//       o0       165     15840         0  
//       o1       165        96         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 96, 165, 165 }
// Out stride: { 1, 15840, 96 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 96, 2, 8 }
// Contraction output var shape: fp32(1, 165, 165, 96):(2613600, 15840, 96, 1):10209.4 KiB
// Computed true ops: 5227200
// Computed work groups: 1743
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 6144
// Computed mem read: 128
// Computed mem write: 6144
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 83, 1
__kernel void kernel_c42_sdk_82(__global float* restrict  X_T260)
{
  int tid = get_local_id(0);
  float agg[6] = {0, 0, 0, 0, 0, 0, };
  int o1_gid = (get_group_id(0) * 8);
  int o0_gid = (get_group_id(1) * 2);
  {
    int c_tid = (tid % 32);
    int o1_tid = ((tid / 32) % 4);
    int o0_tid = ((tid / 128) % 2);
    for (int o1_lid = 0; o1_lid < 2; o1_lid += 1)
    {
      int o1 = ((4 * o1_lid) + o1_tid);
      for (int c_lid = 0; c_lid < 3; c_lid += 1)
      {
        int c = ((32 * c_lid) + c_tid);
        float val1 = 1.0f;
        int agg_idx = (c_lid + (o1_lid * 3));
        agg[agg_idx] = val1;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o1_tid = ((tid / 32) % 4);
  int o0_tid = ((tid / 128) % 2);
  for (int o1_lid = 0; o1_lid < 2; o1_lid += 1)
  {
    int o1_cond = ((o1_lid < 1) || ((o1_gid != 160) || (o1_tid < 1)));
    if (o1_cond)
    {
      int o1 = ((4 * o1_lid) + o1_tid);
      int o0_cond = ((o0_gid != 164) || (o0_tid < 1));
      if (o0_cond)
      {
        for (int c_lid = 0; c_lid < 3; c_lid += 1)
        {
          int c = ((32 * c_lid) + c_tid);
          float LX_T260 = agg[(c_lid + (o1_lid * 3))];
          int gout_idx = ((c + (15840 * (o0_gid + o0_tid))) + (96 * (o1_gid + o1)));
          X_T260[gout_idx] = LX_T260;
        }
      }
    }
  }
}
