#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 4 1
// lid: 256 1 1
// Original:
// X_T651[n0, n1, n2, 352 + a : _T903, _T904, _T905, _T906] = =(X_T650[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T651[n0, n1, n2, 352 + a : _T903, _T904, _T905, _T906] = =(X_T650[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 32, 0 <= 352 + a < 384, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 32 }
// Defracted:
// X_T651[n0, n1, n2, 352 + a : _T903, _T904, _T905, _T906] = =(X_T650[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T651    X_T650  
//        a        32         1         1  
//       n1        14      5376       448  
//       n2        14       384        32  
//      off                 352         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 32, 14, 14 }
// Out stride: { 1, 5376, 384 }
// Input 1 offset: 0
// Input 1 stride: { 1, 448, 32 }
// Tile size: { 32, 2, 4 }
// Contraction output var shape: fp32(1, 14, 14, 384):(75264, 5376, 384, 1):294 KiB
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
__kernel void kernel_c68_sdk_214(__global float* restrict  X_T651, __global const float* restrict  in1)
{
  X_T651 = (X_T651 + 352);
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
    float LX_T651 = agg[0];
    int gout_idx = ((a_tid + (5376 * (n1_gid + n1_tid))) + (384 * (n2_gid + n2_tid)));
    X_T651[gout_idx] = LX_T651;
  }
}
