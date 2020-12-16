#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Original:
// X_T482[n, o0, o1, c : _T719, _T720, _T721, _T722] = =(X_T97[])
// With Index Variables Made Integral:
// X_T482[n, o0, o1, c : _T719, _T720, _T721, _T722] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= c < 44, 0 <= o0 < 56, 0 <= o1 < 56, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= c < 44, 0 <= o0 < 56, 0 <= o1 < 56 }
// Defracted:
// X_T482[n, o0, o1, c : _T719, _T720, _T721, _T722] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range    X_T482     X_T97  
//        c        44         1         0  
//       o0        56      2464         0  
//       o1        56        44         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 44, 56, 56 }
// Out stride: { 1, 2464, 44 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 44, 8, 8 }
// Contraction output var shape: fp32(1, 56, 56, 44):(137984, 2464, 44, 1):539 KiB
// Computed true ops: 275968
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
__kernel void kernel_c42_sdk_167(__global float* restrict  X_T482)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  int o1_gid = (get_group_id(0) * 8);
  int o0_gid = (get_group_id(1) * 8);
  {
    int c_tid = (tid % 32);
    int o1_tid = ((tid / 32) % 4);
    int o0_tid = ((tid / 128) % 2);
    for (int c_lid = 0; c_lid < 2; c_lid += 1)
    {
      int c_cond = ((c_lid < 1) || (c_tid < 12));
      int c = select((int)0, (int)((32 * c_lid) + c_tid), (int)c_cond);
      for (int o1_lid = 0; o1_lid < 2; o1_lid += 1)
      {
        int o1 = ((4 * o1_lid) + o1_tid);
        for (int o0_lid = 0; o0_lid < 4; o0_lid += 1)
        {
          int o0 = ((2 * o0_lid) + o0_tid);
          float val1 = 1.0f;
          int agg_idx = ((c_lid + (o1_lid * 2)) + (o0_lid * 4));
          agg[agg_idx] = select((float)agg[agg_idx], (float)val1, (int)c_cond);
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o1_tid = ((tid / 32) % 4);
  int o0_tid = ((tid / 128) % 2);
  for (int c_lid = 0; c_lid < 2; c_lid += 1)
  {
    int c_cond = ((c_lid < 1) || (c_tid < 12));
    if (c_cond)
    {
      int c = ((32 * c_lid) + c_tid);
      for (int o1_lid = 0; o1_lid < 2; o1_lid += 1)
      {
        int o1 = ((4 * o1_lid) + o1_tid);
        for (int o0_lid = 0; o0_lid < 4; o0_lid += 1)
        {
          int o0 = ((2 * o0_lid) + o0_tid);
          float LX_T482 = agg[((c_lid + (o1_lid * 2)) + (o0_lid * 4))];
          int gout_idx = ((c + (2464 * (o0_gid + o0))) + (44 * (o1_gid + o1)));
          X_T482[gout_idx] = LX_T482;
        }
      }
    }
  }
}
