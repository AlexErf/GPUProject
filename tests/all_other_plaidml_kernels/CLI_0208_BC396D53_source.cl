#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 9 1
// lid: 256 1 1
// Original:
// X_T1415[n, o0, o1, c : _T2220, _T2221, _T2222, _T2223] = =(X_T97[])
// With Index Variables Made Integral:
// X_T1415[n, o0, o1, c : _T2220, _T2221, _T2222, _T2223] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 28, 0 <= o1 < 28, 0 <= c < 264, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 28, 0 <= o1 < 28, 0 <= c < 264 }
// Defracted:
// X_T1415[n, o0, o1, c : _T2220, _T2221, _T2222, _T2223] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range   X_T1415     X_T97  
//        c       264         1         0  
//       o0        28      7392         0  
//       o1        28       264         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 264, 28, 28 }
// Out stride: { 1, 7392, 264 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 32, 4, 28 }
// Contraction output var shape: fp32(1, 28, 28, 264):(206976, 7392, 264, 1):808.5 KiB
// Computed true ops: 413952
// Computed work groups: 63
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 14336
// Computed mem read: 128
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 9, 1
__kernel void kernel_c42_sdk_533(__global float* restrict  X_T1415)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  int c_gid = (get_group_id(1) * 32);
  int o0_gid = (get_group_id(0) * 4);
  {
    int c_tid = (tid % 32);
    int o1_tid = ((tid / 32) % 4);
    int o0_tid = ((tid / 128) % 2);
    for (int o1_lid = 0; o1_lid < 7; o1_lid += 1)
    {
      int o1 = ((4 * o1_lid) + o1_tid);
      for (int o0_lid = 0; o0_lid < 2; o0_lid += 1)
      {
        int o0 = ((2 * o0_lid) + o0_tid);
        float val1 = 1.0f;
        int agg_idx = (o1_lid + (o0_lid * 7));
        agg[agg_idx] = val1;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o1_tid = ((tid / 32) % 4);
  int o0_tid = ((tid / 128) % 2);
  int c_cond = ((c_gid != 256) || (c_tid < 8));
  if (c_cond)
  {
    for (int o1_lid = 0; o1_lid < 7; o1_lid += 1)
    {
      int o1 = ((4 * o1_lid) + o1_tid);
      for (int o0_lid = 0; o0_lid < 2; o0_lid += 1)
      {
        int o0 = ((2 * o0_lid) + o0_tid);
        float LX_T1415 = agg[(o1_lid + (o0_lid * 7))];
        int gout_idx = (((c_gid + c_tid) + (7392 * (o0_gid + o0))) + (264 * o1));
        X_T1415[gout_idx] = LX_T1415;
      }
    }
  }
}
