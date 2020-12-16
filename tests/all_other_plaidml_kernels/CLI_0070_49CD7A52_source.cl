#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 14080 83 1
// lid: 256 1 1
// Original:
// X_T59[n, d0, d1, c : _T36, _T37, _T38, _T39] = =(X_T57[n, -1 + d0, -1 + d1, c])
// With Index Variables Made Integral:
// X_T59[n, d0, d1, c : _T36, _T37, _T38, _T39] = =(X_T57[n, -1 + d0, -1 + d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= c < 42, 0 <= c < 42, 0 <= -1 + d0 < 165, 0 <= -1 + d1 < 165, 0 <= d0 < 167, 0 <= d1 < 167, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= c < 42, 0 <= -1 + d0 < 165, 0 <= -1 + d1 < 165 }
// Defracted:
// X_T59[n, d0, d1, c : _T36, _T37, _T38, _T39] = =(X_T57[n, -1 + d0, -1 + d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range     X_T59     X_T57  
//        c        42         1         1  
//       d0       165      7014      6930  
//       d1       165        42        42  
//      off                7056         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 6930, 165 }
// Out stride: { 1, 7014 }
// Input 1 offset: 0
// Input 1 stride: { 1, 6930 }
// Tile size: { 128, 2 }
// Contraction output var shape: fp32(1, 167, 167, 42):(1171338, 7014, 42, 1):4575.54 KiB
// Computed true ops: 2286900
// Computed work groups: 4565
// Computed inner loops: 1
// Computed shared mem: 1032
// Computed out regs: 1024
// Computed mem read: 1024
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 14080, 83, 1
__kernel void kernel_c42_sdk_6(__global float* restrict  X_T59, __global const float* restrict  in1)
{
  X_T59 = (X_T59 + 7056);
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[258];
  int d1_c_gid = (get_group_id(0) * 128);
  int d0_gid = (get_group_id(1) * 2);
  {
    {
      int gbase = (d1_c_gid + (d0_gid * 6930));
      int d1_c_tid = (tid % 128);
      int d0_tid = ((tid / 128) % 2);
      int lidx = (d1_c_tid + (129 * d0_tid));
      int gidx = ((gbase + d1_c_tid) + (6930 * d0_tid));
      in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)1143449)];
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
  int d1_c_cond = ((d1_c_gid != 6912) || (d1_c_tid < 18));
  if (d1_c_cond)
  {
    int d0_cond = ((d0_gid != 164) || (d0_tid < 1));
    if (d0_cond)
    {
      float LX_T59 = agg[0];
      int gout_idx = ((d1_c_gid + d1_c_tid) + (7014 * (d0_gid + d0_tid)));
      X_T59[gout_idx] = LX_T59;
    }
  }
}
