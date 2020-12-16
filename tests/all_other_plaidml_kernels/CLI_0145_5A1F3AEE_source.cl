#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 21 1
// lid: 256 1 1
// Original:
// X_T482[n, o0, o1, c : _T719, _T720, _T721, _T722] = =(X_T100[])
// With Index Variables Made Integral:
// X_T482[n, o0, o1, c : _T719, _T720, _T721, _T722] = =(X_T100[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 83, 0 <= o1 < 83, 0 <= c < 168, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 83, 0 <= o1 < 83, 0 <= c < 168 }
// Defracted:
// X_T482[n, o0, o1, c : _T719, _T720, _T721, _T722] = =(X_T100[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range    X_T482    X_T100  
//        c       168         1         0  
//       o0        83     13944         0  
//       o1        83       168         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 168, 83, 83 }
// Out stride: { 1, 13944, 168 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 168, 4, 4 }
// Contraction output var shape: fp32(1, 83, 83, 168):(1157352, 13944, 168, 1):4520.91 KiB
// Computed true ops: 2314704
// Computed work groups: 441
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 12288
// Computed mem read: 128
// Computed mem write: 12288
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 21, 1
__kernel void kernel_c42_sdk_167(__global float* restrict  X_T482)
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
        int c_cond = ((c_lid < 5) || (c_tid < 8));
        int c = select((int)0, (int)((32 * c_lid) + c_tid), (int)c_cond);
        float val1 = 1.0f;
        int agg_idx = (c_lid + (o0_lid * 6));
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
        for (int c_lid = 0; c_lid < 6; c_lid += 1)
        {
          int c_cond = ((c_lid < 5) || (c_tid < 8));
          if (c_cond)
          {
            int c = ((32 * c_lid) + c_tid);
            float LX_T482 = agg[(c_lid + (o0_lid * 6))];
            int gout_idx = ((c + (13944 * (o0_gid + o0))) + (168 * (o1_gid + o1_tid)));
            X_T482[gout_idx] = LX_T482;
          }
        }
      }
    }
  }
}
