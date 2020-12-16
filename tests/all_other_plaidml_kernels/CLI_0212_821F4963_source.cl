#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 331 1
// lid: 256 1 1
// Original:
// X_T1773[n, d0, d1, c : _T2797, _T2798, _T2799, _T2800] = =(X_T1385[n, d0, d1, c])
// With Index Variables Made Integral:
// X_T1773[n, d0, d1, c : _T2797, _T2798, _T2799, _T2800] = =(X_T1385[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= d0 < 42, 0 <= d1 < 42, 0 <= d0 < 43, 0 <= d1 < 43, 0 <= c < 1008, 0 <= c < 1008, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= d0 < 42, 0 <= d1 < 42, 0 <= c < 1008 }
// Defracted:
// X_T1773[n, d0, d1, c : _T2797, _T2798, _T2799, _T2800] = =(X_T1385[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range   X_T1773   X_T1385  
//        c      1008         1         1  
//       d0        42     43344     42336  
//       d1        42      1008      1008  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 42336, 42 }
// Out stride: { 1, 43344 }
// Input 1 offset: 0
// Input 1 stride: { 1, 42336 }
// Tile size: { 128, 2 }
// Contraction output var shape: fp32(1, 43, 43, 1008):(1863792, 43344, 1008, 1):7280.44 KiB
// Computed true ops: 3556224
// Computed work groups: 6951
// Computed inner loops: 1
// Computed shared mem: 1032
// Computed out regs: 1024
// Computed mem read: 1024
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 331, 1
__kernel void kernel_c42_sdk_675(__global float* restrict  X_T1773, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[258];
  int d1_c_gid = (get_group_id(1) * 128);
  int d0_gid = (get_group_id(0) * 2);
  {
    {
      int gbase = (d1_c_gid + (d0_gid * 42336));
      int d1_c_tid = (tid % 128);
      int d0_tid = ((tid / 128) % 2);
      int lidx = (d1_c_tid + (129 * d0_tid));
      int gidx = ((gbase + d1_c_tid) + (42336 * (int)d0_tid));
      in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)1778111)];
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
  int d1_c_cond = ((d1_c_gid != 42240) || (d1_c_tid < 96));
  if (d1_c_cond)
  {
    float LX_T1773 = agg[0];
    int gout_idx = ((d1_c_gid + d1_c_tid) + (43344 * (int)(d0_gid + d0_tid)));
    X_T1773[gout_idx] = LX_T1773;
  }
}
