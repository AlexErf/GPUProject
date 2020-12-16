#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Original:
// X_T105[n0, n1, n2, 64 + a : _T63, _T64, _T65, _T66] = =(X_T104[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T105[n0, n1, n2, 64 + a : _T63, _T64, _T65, _T66] = =(X_T104[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= a < 32, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= 64 + a < 96, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= a < 32, 0 <= n1 < 56, 0 <= n2 < 56 }
// Defracted:
// X_T105[n0, n1, n2, 64 + a : _T63, _T64, _T65, _T66] = =(X_T104[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T105    X_T104  
//        a        32         1         1  
//       n1        56      5376      1792  
//       n2        56        96        32  
//      off                  64         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 32, 56, 56 }
// Out stride: { 1, 5376, 96 }
// Input 1 offset: 0
// Input 1 stride: { 1, 1792, 32 }
// Tile size: { 32, 56, 2 }
// Contraction output var shape: fp32(1, 56, 56, 96):(301056, 5376, 96, 1):1176 KiB
// Computed true ops: 200704
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 14560
// Computed out regs: 14336
// Computed mem read: 14336
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c108_sdk_13(__global float* restrict  X_T105, __global const float* restrict  in1)
{
  X_T105 = (X_T105 + 64);
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3640];
  int n2_gid = (get_group_id(0) * 2);
  {
    {
      int gbase = (n2_gid * 32);
      int a_n2_tid = (tid % 64);
      int n1_tid = ((tid / 64) % 4);
      for (int n1_lid = 0; n1_lid < 14; n1_lid += 1)
      {
        int n1 = ((4 * n1_lid) + n1_tid);
        int lidx = (a_n2_tid + (65 * n1));
        int gidx = ((gbase + a_n2_tid) + (1792 * n1));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)100351)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 2);
    int n1_tid = ((tid / 64) % 4);
    for (int n1_lid = 0; n1_lid < 14; n1_lid += 1)
    {
      int n1 = ((4 * n1_lid) + n1_tid);
      float val1 = in1_shared[((a_tid + (32 * n2_tid)) + (65 * n1))];
      agg[n1_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 2);
  int n1_tid = ((tid / 64) % 4);
  for (int n1_lid = 0; n1_lid < 14; n1_lid += 1)
  {
    int n1 = ((4 * n1_lid) + n1_tid);
    float LX_T105 = agg[n1_lid];
    int gout_idx = ((a_tid + (5376 * n1)) + (96 * (n2_gid + n2_tid)));
    X_T105[gout_idx] = LX_T105;
  }
}
