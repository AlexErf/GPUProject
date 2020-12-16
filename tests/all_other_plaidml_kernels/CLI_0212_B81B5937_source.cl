#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2560 2 1
// lid: 256 1 1
// Original:
// X_T993[n, o0, o1, c : _T1408, _T1409, _T1410, _T1411] = =(X_T160[])
// With Index Variables Made Integral:
// X_T993[n, o0, o1, c : _T1408, _T1409, _T1410, _T1411] = =(X_T160[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 8, 0 <= o1 < 8, 0 <= c < 1280, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 8, 0 <= o1 < 8, 0 <= c < 1280 }
// Defracted:
// X_T993[n, o0, o1, c : _T1408, _T1409, _T1410, _T1411] = =(X_T160[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range    X_T993    X_T160  
//        c      1280         1         0  
//       o0         8     10240         0  
//       o1         8      1280         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 1280, 8, 8 }
// Out stride: { 1, 10240, 1280 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 128, 4, 8 }
// Contraction output var shape: fp32(1, 8, 8, 1280):(81920, 10240, 1280, 1):320 KiB
// Computed true ops: 163840
// Computed work groups: 20
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 16384
// Computed mem read: 128
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2560, 2, 1
__kernel void kernel_c56_sdk_342(__global float* restrict  X_T993)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  int c_gid = (get_group_id(0) * 128);
  int o0_gid = (get_group_id(1) * 4);
  {
    int c_tid = (tid % 32);
    int o1_tid = ((tid / 32) % 4);
    int o0_tid = ((tid / 128) % 2);
    for (int c_lid = 0; c_lid < 4; c_lid += 1)
    {
      int c = ((32 * c_lid) + c_tid);
      for (int o1_lid = 0; o1_lid < 2; o1_lid += 1)
      {
        int o1 = ((4 * o1_lid) + o1_tid);
        for (int o0_lid = 0; o0_lid < 2; o0_lid += 1)
        {
          int o0 = ((2 * o0_lid) + o0_tid);
          float val1 = 1.0f;
          int agg_idx = ((c_lid + (o1_lid * 4)) + (o0_lid * 8));
          agg[agg_idx] = val1;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o1_tid = ((tid / 32) % 4);
  int o0_tid = ((tid / 128) % 2);
  for (int c_lid = 0; c_lid < 4; c_lid += 1)
  {
    int c = ((32 * c_lid) + c_tid);
    for (int o1_lid = 0; o1_lid < 2; o1_lid += 1)
    {
      int o1 = ((4 * o1_lid) + o1_tid);
      for (int o0_lid = 0; o0_lid < 2; o0_lid += 1)
      {
        int o0 = ((2 * o0_lid) + o0_tid);
        float LX_T993 = agg[((c_lid + (o1_lid * 4)) + (o0_lid * 8))];
        int gout_idx = (((c_gid + c) + (10240 * (o0_gid + o0))) + (1280 * o1));
        X_T993[gout_idx] = LX_T993;
      }
    }
  }
}
