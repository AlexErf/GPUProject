#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 56 1
// lid: 256 1 1
// Original:
// X_T107[n0, n1, n2, a : _T68, _T69, _T70, _T71] = =(X_T106[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T107[n0, n1, n2, a : _T68, _T69, _T70, _T71] = =(X_T106[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= a < 96, 0 <= a < 128, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= a < 96 }
// Defracted:
// X_T107[n0, n1, n2, a : _T68, _T69, _T70, _T71] = =(X_T106[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T107    X_T106  
//        a        96         1         1  
//       n1        56      7168      5376  
//       n2        56       128        96  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 96, 56, 56 }
// Out stride: { 1, 7168, 128 }
// Input 1 offset: 0
// Input 1 stride: { 1, 5376, 96 }
// Tile size: { 96, 8, 1 }
// Contraction output var shape: fp32(1, 56, 56, 128):(401408, 7168, 128, 1):1568 KiB
// Computed true ops: 602112
// Computed work groups: 392
// Computed inner loops: 1
// Computed shared mem: 3104
// Computed out regs: 3072
// Computed mem read: 3072
// Computed mem write: 3072
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 56, 1
__kernel void kernel_c108_sdk_15(__global float* restrict  X_T107, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[3] = {0, 0, 0, };
  __local float in1_shared[776];
  int n2_gid = get_group_id(1);
  int n1_gid = (get_group_id(0) * 8);
  {
    {
      int gbase = ((n2_gid * 96) + (n1_gid * 5376));
      int a_n2_tid = (tid % 128);
      int n1_tid = ((tid / 128) % 2);
      int a_n2_cond = (a_n2_tid < 96);
      if (a_n2_cond)
      {
        for (int n1_lid = 0; n1_lid < 4; n1_lid += 1)
        {
          int n1 = ((2 * n1_lid) + n1_tid);
          int lidx = (a_n2_tid + (97 * n1));
          int gidx = ((gbase + a_n2_tid) + (5376 * n1));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)301055)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n1_tid = ((tid / 32) % 8);
    for (int a_lid = 0; a_lid < 3; a_lid += 1)
    {
      int a = ((32 * a_lid) + a_tid);
      float val1 = in1_shared[(a + (97 * n1_tid))];
      agg[a_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n1_tid = ((tid / 32) % 8);
  for (int a_lid = 0; a_lid < 3; a_lid += 1)
  {
    int a = ((32 * a_lid) + a_tid);
    float LX_T107 = agg[a_lid];
    int gout_idx = ((a + (7168 * (n1_gid + n1_tid))) + (128 * n2_gid));
    X_T107[gout_idx] = LX_T107;
  }
}
