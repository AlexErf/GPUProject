#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 111 1
// lid: 256 1 1
// Original:
// X_T1558[n, d0, d1, c : _T2451, _T2452, _T2453, _T2454] = =(X_T1556[n, d0, d1, c])
// With Index Variables Made Integral:
// X_T1558[n, d0, d1, c : _T2451, _T2452, _T2453, _T2454] = =(X_T1556[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= d0 < 42, 0 <= d1 < 42, 0 <= d0 < 43, 0 <= d1 < 43, 0 <= c < 336, 0 <= c < 336, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= d0 < 42, 0 <= d1 < 42, 0 <= c < 336 }
// Defracted:
// X_T1558[n, d0, d1, c : _T2451, _T2452, _T2453, _T2454] = =(X_T1556[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range   X_T1558   X_T1556  
//        c       336         1         1  
//       d0        42     14448     14112  
//       d1        42       336       336  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 14112, 42 }
// Out stride: { 1, 14448 }
// Input 1 offset: 0
// Input 1 stride: { 1, 14112 }
// Tile size: { 128, 2 }
// Contraction output var shape: fp32(1, 43, 43, 336):(621264, 14448, 336, 1):2426.81 KiB
// Computed true ops: 1185408
// Computed work groups: 2331
// Computed inner loops: 1
// Computed shared mem: 1032
// Computed out regs: 1024
// Computed mem read: 1024
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 111, 1
__kernel void kernel_c42_sdk_593(__global float* restrict  X_T1558, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[258];
  int d1_c_gid = (get_group_id(1) * 128);
  int d0_gid = (get_group_id(0) * 2);
  {
    {
      int gbase = (d1_c_gid + (d0_gid * 14112));
      int d1_c_tid = (tid % 128);
      int d0_tid = ((tid / 128) % 2);
      int lidx = (d1_c_tid + (129 * d0_tid));
      int gidx = ((gbase + d1_c_tid) + (14112 * d0_tid));
      in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)592703)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int d1_c_tid = (tid % 128);
    int d0_tid = ((tid / 128) % 2);
    float val1 = in1_shared[(d1_c_tid + (129 * d0_tid))];
    agg[0] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int d1_c_tid = (tid % 128);
  int d0_tid = ((tid / 128) % 2);
  int d1_c_cond = ((d1_c_gid != 14080) || (d1_c_tid < 32));
  if (d1_c_cond)
  {
    float LX_T1558 = agg[0];
    int gout_idx = ((d1_c_gid + d1_c_tid) + (14448 * (d0_gid + d0_tid)));
    X_T1558[gout_idx] = LX_T1558;
  }
}
