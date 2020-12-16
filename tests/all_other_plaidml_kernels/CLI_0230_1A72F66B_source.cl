#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 21 1
// lid: 256 1 1
// Original:
// X_T2832[n, d0, d1, c : _T4499, _T4500, _T4501, _T4502] = =(X_T2831[n, -1 + d0, -1 + d1, c])
// With Index Variables Made Integral:
// X_T2832[n, d0, d1, c : _T4499, _T4500, _T4501, _T4502] = =(X_T2831[n, -1 + d0, -1 + d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= -1 + d0 < 21, 0 <= -1 + d1 < 21, 0 <= d0 < 23, 0 <= d1 < 23, 0 <= c < 672, 0 <= c < 672, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= -1 + d0 < 21, 0 <= -1 + d1 < 21, 0 <= c < 672 }
// Defracted:
// X_T2832[n, d0, d1, c : _T4499, _T4500, _T4501, _T4502] = =(X_T2831[n, -1 + d0, -1 + d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range   X_T2832   X_T2831  
//        c       672         1         1  
//       d0        21     15456     14112  
//       d1        21       672       672  
//      off               16128         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 14112, 21 }
// Out stride: { 1, 15456 }
// Input 1 offset: 0
// Input 1 stride: { 1, 14112 }
// Tile size: { 2048, 1 }
// Contraction output var shape: fp32(1, 23, 23, 672):(355488, 15456, 672, 1):1388.62 KiB
// Computed true ops: 592704
// Computed work groups: 147
// Computed inner loops: 1
// Computed shared mem: 8192
// Computed out regs: 8192
// Computed mem read: 8192
// Computed mem write: 8192
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 21, 1
__kernel void kernel_c42_sdk_1095(__global float* restrict  X_T2832, __global const float* restrict  in1)
{
  X_T2832 = (X_T2832 + 16128);
  int tid = get_local_id(0);
  float agg[8] = {0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[2048];
  int d1_c_gid = (get_group_id(0) * 2048);
  int d0_gid = get_group_id(1);
  {
    {
      int gbase = (d1_c_gid + (d0_gid * 14112));
      int d1_c_tid = (tid % 256);
      for (int d1_c_lid = 0; d1_c_lid < 8; d1_c_lid += 1)
      {
        int d1_c = ((256 * d1_c_lid) + d1_c_tid);
        int gidx = (gbase + d1_c);
        in1_shared[d1_c] = in1[clamp((int)gidx, (int)0, (int)296351)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int d1_c_tid = (tid % 256);
    for (int d1_c_lid = 0; d1_c_lid < 8; d1_c_lid += 1)
    {
      int d1_c = ((256 * d1_c_lid) + d1_c_tid);
      float val1 = in1_shared[d1_c];
      agg[d1_c_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int d1_c_tid = (tid % 256);
  for (int d1_c_lid = 0; d1_c_lid < 8; d1_c_lid += 1)
  {
    int d1_c_cond = ((d1_c_lid < 7) || ((d1_c_gid != 12288) || (d1_c_tid < 32)));
    if (d1_c_cond)
    {
      int d1_c = ((256 * d1_c_lid) + d1_c_tid);
      float LX_T2832 = agg[d1_c_lid];
      int gout_idx = ((d1_c_gid + d1_c) + (15456 * d0_gid));
      X_T2832[gout_idx] = LX_T2832;
    }
  }
}
