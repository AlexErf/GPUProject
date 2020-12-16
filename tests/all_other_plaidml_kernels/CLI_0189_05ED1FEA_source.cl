#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2816 11 11
// lid: 256 1 1
// Original:
// X_T1607[n, o0, o1, c : _T2531, _T2532, _T2533, _T2534] = =(X_T100[])
// With Index Variables Made Integral:
// X_T1607[n, o0, o1, c : _T2531, _T2532, _T2533, _T2534] = =(X_T100[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 43, 0 <= o1 < 43, 0 <= c < 336, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 43, 0 <= o1 < 43, 0 <= c < 336 }
// Defracted:
// X_T1607[n, o0, o1, c : _T2531, _T2532, _T2533, _T2534] = =(X_T100[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range   X_T1607    X_T100  
//        c       336         1         0  
//       o0        43     14448         0  
//       o1        43       336         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 336, 43, 43 }
// Out stride: { 1, 14448, 336 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 32, 4, 4 }
// Contraction output var shape: fp32(1, 43, 43, 336):(621264, 14448, 336, 1):2426.81 KiB
// Computed true ops: 1242528
// Computed work groups: 1331
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 2048
// Computed mem read: 128
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2816, 11, 11
__kernel void kernel_c42_sdk_611(__global float* restrict  X_T1607)
{
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  int c_gid = (get_group_id(0) * 32);
  int o1_gid = (get_group_id(1) * 4);
  int o0_gid = (get_group_id(2) * 4);
  {
    int c_tid = (tid % 32);
    int o1_tid = ((tid / 32) % 4);
    int o0_tid = ((tid / 128) % 2);
    for (int o0_lid = 0; o0_lid < 2; o0_lid += 1)
    {
      int o0 = ((2 * o0_lid) + o0_tid);
      float val1 = 1.0f;
      agg[o0_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o1_tid = ((tid / 32) % 4);
  int o0_tid = ((tid / 128) % 2);
  int c_cond = ((c_gid != 320) || (c_tid < 16));
  if (c_cond)
  {
    int o1_cond = ((o1_gid != 40) || (o1_tid < 3));
    if (o1_cond)
    {
      for (int o0_lid = 0; o0_lid < 2; o0_lid += 1)
      {
        int o0_cond = ((o0_lid < 1) || ((o0_gid != 40) || (o0_tid < 1)));
        if (o0_cond)
        {
          int o0 = ((2 * o0_lid) + o0_tid);
          float LX_T1607 = agg[o0_lid];
          int gout_idx = (((c_gid + c_tid) + (14448 * (o0_gid + o0))) + (336 * (o1_gid + o1_tid)));
          X_T1607[gout_idx] = LX_T1607;
        }
      }
    }
  }
}
