#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 8 1
// lid: 256 1 1
// Original:
// X_T2047[n0, n1, n2, 192 + a : _T2865, _T2866, _T2867, _T2868] = =(X_T2046[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T2047[n0, n1, n2, 192 + a : _T2865, _T2866, _T2867, _T2868] = =(X_T2046[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 8, 0 <= n2 < 8, 0 <= n1 < 8, 0 <= n2 < 8, 0 <= a < 256, 0 <= 192 + a < 448, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 8, 0 <= n2 < 8, 0 <= a < 256 }
// Defracted:
// X_T2047[n0, n1, n2, 192 + a : _T2865, _T2866, _T2867, _T2868] = =(X_T2046[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T2047   X_T2046  
//        a       256         1         1  
//       n1         8      3584      2048  
//       n2         8       448       256  
//      off                 192         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 256, 8, 8 }
// Out stride: { 1, 3584, 448 }
// Input 1 offset: 0
// Input 1 stride: { 1, 2048, 256 }
// Tile size: { 128, 1, 8 }
// Contraction output var shape: fp32(1, 8, 8, 448):(28672, 3584, 448, 1):112 KiB
// Computed true ops: 32768
// Computed work groups: 16
// Computed inner loops: 1
// Computed shared mem: 4128
// Computed out regs: 4096
// Computed mem read: 4096
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 512, 8, 1
__kernel void kernel_c51_sdk_668(__global float* restrict  X_T2047, __global const float* restrict  in1)
{
  X_T2047 = (X_T2047 + 192);
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[1032];
  int a_gid = (get_group_id(0) * 128);
  int n1_gid = get_group_id(1);
  {
    {
      int gbase = (a_gid + (n1_gid * 2048));
      int a_tid = (tid % 128);
      int n2_n1_tid = ((tid / 128) % 2);
      for (int n2_n1_lid = 0; n2_n1_lid < 4; n2_n1_lid += 1)
      {
        int n2_n1 = ((2 * n2_n1_lid) + n2_n1_tid);
        int lidx = (a_tid + (129 * n2_n1));
        int gidx = ((gbase + a_tid) + (256 * n2_n1));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)16383)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 8);
    for (int a_lid = 0; a_lid < 4; a_lid += 1)
    {
      int a = ((32 * a_lid) + a_tid);
      float val1 = in1_shared[(a + (129 * n2_tid))];
      agg[a_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 8);
  for (int a_lid = 0; a_lid < 4; a_lid += 1)
  {
    int a = ((32 * a_lid) + a_tid);
    float LX_T2047 = agg[a_lid];
    int gout_idx = (((a_gid + a) + (3584 * n1_gid)) + (448 * n2_tid));
    X_T2047[gout_idx] = LX_T2047;
  }
}
