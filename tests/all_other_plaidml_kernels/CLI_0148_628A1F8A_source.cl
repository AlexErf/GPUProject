#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 2
// lid: 256 1 1
// Original:
// X_T226[n, o0, o1, c : _T274, _T275, _T276, _T277] = =(X_T225[])
// With Index Variables Made Integral:
// X_T226[n, o0, o1, c : _T274, _T275, _T276, _T277] = =(X_T225[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 56, 0 <= o1 < 56, 0 <= c < 128, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 56, 0 <= o1 < 56, 0 <= c < 128 }
// Defracted:
// X_T226[n, o0, o1, c : _T274, _T275, _T276, _T277] = =(X_T225[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range    X_T226    X_T225  
//        c       128         1         0  
//       o0        56      7168         0  
//       o1        56       128         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 128, 56, 56 }
// Out stride: { 1, 7168, 128 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 64, 8, 8 }
// Contraction output var shape: fp32(1, 56, 56, 128):(401408, 7168, 128, 1):1568 KiB
// Computed true ops: 802816
// Computed work groups: 98
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 16384
// Computed mem read: 128
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 2
__kernel void kernel_c68_sdk_64(__global float* restrict  X_T226)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  int c_gid = (get_group_id(2) * 64);
  int o1_gid = (get_group_id(0) * 8);
  int o0_gid = (get_group_id(1) * 8);
  {
    int c_tid = (tid % 32);
    int o1_tid = ((tid / 32) % 4);
    int o0_tid = ((tid / 128) % 2);
    for (int c_lid = 0; c_lid < 2; c_lid += 1)
    {
      int c = ((32 * c_lid) + c_tid);
      for (int o1_lid = 0; o1_lid < 2; o1_lid += 1)
      {
        int o1 = ((4 * o1_lid) + o1_tid);
        for (int o0_lid = 0; o0_lid < 4; o0_lid += 1)
        {
          int o0 = ((2 * o0_lid) + o0_tid);
          float val1 = 1.0f;
          int agg_idx = ((c_lid + (o1_lid * 2)) + (o0_lid * 4));
          agg[agg_idx] = val1;
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
    int c = ((32 * c_lid) + c_tid);
    for (int o1_lid = 0; o1_lid < 2; o1_lid += 1)
    {
      int o1 = ((4 * o1_lid) + o1_tid);
      for (int o0_lid = 0; o0_lid < 4; o0_lid += 1)
      {
        int o0 = ((2 * o0_lid) + o0_tid);
        float LX_T226 = agg[((c_lid + (o1_lid * 2)) + (o0_lid * 4))];
        int gout_idx = (((c_gid + c) + (7168 * (o0_gid + o0))) + (128 * (o1_gid + o1)));
        X_T226[gout_idx] = LX_T226;
      }
    }
  }
}
