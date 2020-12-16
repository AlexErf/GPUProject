#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 28 1
// lid: 256 1 1
// Original:
// X_T303[n0, n1, n2, a : _T359, _T360, _T361, _T362] = =(X_T302[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T303[n0, n1, n2, a : _T359, _T360, _T361, _T362] = =(X_T302[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= a < 192, 0 <= a < 224, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= a < 192 }
// Defracted:
// X_T303[n0, n1, n2, a : _T359, _T360, _T361, _T362] = =(X_T302[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T303    X_T302  
//        a       192         1         1  
//       n1        28      6272      5376  
//       n2        28       224       192  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 192, 28, 28 }
// Out stride: { 1, 6272, 224 }
// Input 1 offset: 0
// Input 1 stride: { 1, 5376, 192 }
// Tile size: { 192, 4, 1 }
// Contraction output var shape: fp32(1, 28, 28, 224):(175616, 6272, 224, 1):686 KiB
// Computed true ops: 301056
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 3088
// Computed out regs: 3072
// Computed mem read: 3072
// Computed mem write: 3072
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 28, 1
__kernel void kernel_c108_sdk_84(__global float* restrict  X_T303, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[3] = {0, 0, 0, };
  __local float in1_shared[772];
  int n2_gid = get_group_id(1);
  int n1_gid = (get_group_id(0) * 4);
  {
    {
      int gbase = ((n2_gid * 192) + (n1_gid * 5376));
      int a_n2_tid = (tid % 256);
      int a_n2_cond = (a_n2_tid < 192);
      if (a_n2_cond)
      {
        for (int n1_lid = 0; n1_lid < 4; n1_lid += 1)
        {
          int lidx = (a_n2_tid + (193 * n1_lid));
          int gidx = ((gbase + a_n2_tid) + (5376 * n1_lid));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)150527)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 64);
    int n1_tid = ((tid / 64) % 4);
    for (int a_lid = 0; a_lid < 3; a_lid += 1)
    {
      int a = ((64 * a_lid) + a_tid);
      float val1 = in1_shared[(a + (193 * n1_tid))];
      agg[a_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 64);
  int n1_tid = ((tid / 64) % 4);
  for (int a_lid = 0; a_lid < 3; a_lid += 1)
  {
    int a = ((64 * a_lid) + a_tid);
    float LX_T303 = agg[a_lid];
    int gout_idx = ((a + (6272 * (n1_gid + n1_tid))) + (224 * n2_gid));
    X_T303[gout_idx] = LX_T303;
  }
}
