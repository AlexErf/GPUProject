#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 2 1
// lid: 256 1 1
// Original:
// X_T59[n, x0, x1, c : _T62, _T63, _T64, _T65] = >(X_T58[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2
// With Index Variables Made Integral:
// X_T59[n, x0, x1, c : _T62, _T63, _T64, _T65] = >(X_T58[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 2, 0 <= k1 < 2, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + 2*x0 < 56, 0 <= k1 + 2*x1 < 56, 0 <= c < 256, 0 <= c < 256, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 2, 0 <= k1 < 2, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + 2*x0 < 56, 0 <= k1 + 2*x1 < 56, 0 <= c < 256 }
// Defracted:
// X_T59[n, x0, x1, c : _T62, _T63, _T64, _T65] = >(X_T58[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T59     X_T58  
//        c       256         1         1  
//       k0         2         0     14336  
//       k1         2         0       256  
//       x0        28      7168     28672  
//       x1        28       256       512  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 256, 2, 2, 28, 28 }
// Out stride: { 1, 0, 0, 7168, 256 }
// Input 1 offset: 0
// Input 1 stride: { 1, 14336, 256, 28672, 512 }
// Tile size: { 128, 1, 1, 28, 1 }
// Contraction output var shape: fp32(1, 28, 28, 256):(200704, 7168, 256, 1):784 KiB
// Computed true ops: 1605632
// Computed work groups: 56
// Computed inner loops: 4
// Computed shared mem: 14448
// Computed out regs: 16384
// Computed mem read: 14336
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 2, 1
__kernel void kernel_c18_sdk_9(__global float* restrict  X_T59, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[16] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, };
  __local float in1_shared[3612];
  int c_gid = (get_group_id(1) * 128);
  int x1_gid = get_group_id(0);
  for (int k0_gid = 0; k0_gid < 2; k0_gid += 1)
  {
    for (int k1_gid = 0; k1_gid < 2; k1_gid += 1)
    {
      {
        int gbase = (((c_gid + (k1_gid * 256)) + (x1_gid * 512)) + (k0_gid * 14336));
        int c_tid = (tid % 128);
        int x0_tid = ((tid / 128) % 2);
        for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          int lidx = (c_tid + (129 * x0));
          int gidx = ((gbase + c_tid) + (28672 * x0));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)802815)];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      int c_tid = (tid % 32);
      int x0_tid = ((tid / 32) % 8);
      for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 3) || (x0_tid < 4));
        int x0 = select((int)0, (int)((8 * x0_lid) + x0_tid), (int)x0_cond);
        for (int c_lid = 0; c_lid < 4; c_lid += 1)
        {
          int c = ((32 * c_lid) + c_tid);
          float val1 = in1_shared[(c + (129 * x0))];
          int agg_idx = (c_lid + (x0_lid * 4));
          float agg_rhs = select((float)agg[agg_idx], (float)val1, (int)(val1 > agg[agg_idx]));
          agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)x0_cond);
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 32);
  int x0_tid = ((tid / 32) % 8);
  for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
  {
    int x0_cond = ((x0_lid < 3) || (x0_tid < 4));
    if (x0_cond)
    {
      int x0 = ((8 * x0_lid) + x0_tid);
      for (int c_lid = 0; c_lid < 4; c_lid += 1)
      {
        int c = ((32 * c_lid) + c_tid);
        float LX_T59 = agg[(c_lid + (x0_lid * 4))];
        LX_T59 = select((float)LX_T59, (float)0, (int)(LX_T59 == (float)-FLT_MAX));
        int gout_idx = (((c_gid + c) + (7168 * x0)) + (256 * x1_gid));
        X_T59[gout_idx] = LX_T59;
      }
    }
  }
}
