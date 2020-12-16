#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 14336 1 1
// lid: 256 1 1
// Original:
// X_T77[n0, n1, n2, a : _T31, _T32, _T33, _T34] = =(X_T76[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T77[n0, n1, n2, a : _T31, _T32, _T33, _T34] = =(X_T76[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= a < 64, 0 <= a < 96, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= a < 64 }
// Defracted:
// X_T77[n0, n1, n2, a : _T31, _T32, _T33, _T34] = =(X_T76[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range     X_T77     X_T76  
//        a        64         1         1  
//       n1        56      5376      3584  
//       n2        56        96        64  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 64, 56, 56 }
// Out stride: { 1, 5376, 96 }
// Input 1 offset: 0
// Input 1 stride: { 1, 3584, 64 }
// Tile size: { 64, 56, 1 }
// Contraction output var shape: fp32(1, 56, 56, 96):(301056, 5376, 96, 1):1176 KiB
// Computed true ops: 401408
// Computed work groups: 56
// Computed inner loops: 1
// Computed shared mem: 14560
// Computed out regs: 14336
// Computed mem read: 14336
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 14336, 1, 1
__kernel void kernel_c108_sdk_6(__global float* restrict  X_T77, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3640];
  int n2_gid = get_group_id(0);
  {
    {
      int gbase = (n2_gid * 64);
      int a_n2_tid = (tid % 64);
      int n1_tid = ((tid / 64) % 4);
      for (int n1_lid = 0; n1_lid < 14; n1_lid += 1)
      {
        int n1 = ((4 * n1_lid) + n1_tid);
        int lidx = (a_n2_tid + (65 * n1));
        int gidx = ((gbase + a_n2_tid) + (3584 * n1));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)200703)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n1_tid = ((tid / 32) % 8);
    for (int a_lid = 0; a_lid < 2; a_lid += 1)
    {
      int a = ((32 * a_lid) + a_tid);
      for (int n1_lid = 0; n1_lid < 7; n1_lid += 1)
      {
        int n1 = ((8 * n1_lid) + n1_tid);
        float val1 = in1_shared[(a + (65 * n1))];
        int agg_idx = (a_lid + (n1_lid * 2));
        agg[agg_idx] = val1;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n1_tid = ((tid / 32) % 8);
  for (int a_lid = 0; a_lid < 2; a_lid += 1)
  {
    int a = ((32 * a_lid) + a_tid);
    for (int n1_lid = 0; n1_lid < 7; n1_lid += 1)
    {
      int n1 = ((8 * n1_lid) + n1_tid);
      float LX_T77 = agg[(a_lid + (n1_lid * 2))];
      int gout_idx = ((a + (5376 * n1)) + (96 * n2_gid));
      X_T77[gout_idx] = LX_T77;
    }
  }
}
