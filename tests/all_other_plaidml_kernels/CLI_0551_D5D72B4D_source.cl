#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 4 1
// lid: 256 1 1
// Original:
// X_T1629[n0, n1, n2, 1568 + a : _T2309, _T2310, _T2311, _T2312] = =(X_T1628[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T1629[n0, n1, n2, 1568 + a : _T2309, _T2310, _T2311, _T2312] = =(X_T1628[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 32, 0 <= 1568 + a < 1600, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 32 }
// Defracted:
// X_T1629[n0, n1, n2, 1568 + a : _T2309, _T2310, _T2311, _T2312] = =(X_T1628[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T1629   X_T1628  
//        a        32         1         1  
//       n1        14     22400       448  
//       n2        14      1600        32  
//      off                1568         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 32, 14, 14 }
// Out stride: { 1, 22400, 1600 }
// Input 1 offset: 0
// Input 1 stride: { 1, 448, 32 }
// Tile size: { 32, 2, 4 }
// Contraction output var shape: fp32(1, 14, 14, 1600):(313600, 22400, 1600, 1):1225 KiB
// Computed true ops: 12544
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 1032
// Computed out regs: 1024
// Computed mem read: 1024
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 4, 1
__kernel void kernel_c124_sdk_556(__global float* restrict  X_T1629, __global const float* restrict  in1)
{
  X_T1629 = (X_T1629 + 1568);
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[258];
  int n2_gid = (get_group_id(1) * 4);
  int n1_gid = (get_group_id(0) * 2);
  {
    {
      int gbase = ((n2_gid * 32) + (n1_gid * 448));
      int a_n2_tid = (tid % 128);
      int n1_tid = ((tid / 128) % 2);
      int lidx = (a_n2_tid + (129 * n1_tid));
      int gidx = ((gbase + a_n2_tid) + (448 * n1_tid));
      in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)6271)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 4);
    int n1_tid = ((tid / 128) % 2);
    float val1 = in1_shared[((a_tid + (32 * n2_tid)) + (129 * n1_tid))];
    agg[0] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 4);
  int n1_tid = ((tid / 128) % 2);
  int n2_cond = ((n2_gid != 12) || (n2_tid < 2));
  if (n2_cond)
  {
    float LX_T1629 = agg[0];
    int gout_idx = ((a_tid + (22400 * (n1_gid + n1_tid))) + (1600 * (n2_gid + n2_tid)));
    X_T1629[gout_idx] = LX_T1629;
  }
}
