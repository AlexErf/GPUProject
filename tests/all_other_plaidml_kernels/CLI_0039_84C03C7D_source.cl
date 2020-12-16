#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 2 1
// lid: 256 1 1
// Original:
// X_T80[n, x0, x1, c : _T103, _T104, _T105, _T106] = >(X_T79[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2
// With Index Variables Made Integral:
// X_T80[n, x0, x1, c : _T103, _T104, _T105, _T106] = >(X_T79[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 2, 0 <= k1 < 2, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + 2*x0 < 28, 0 <= k1 + 2*x1 < 28, 0 <= c < 512, 0 <= c < 512, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 2, 0 <= k1 < 2, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + 2*x0 < 28, 0 <= k1 + 2*x1 < 28, 0 <= c < 512 }
// Defracted:
// X_T80[n, x0, x1, c : _T103, _T104, _T105, _T106] = >(X_T79[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T80     X_T79  
//        c       512         1         1  
//       k0         2         0     14336  
//       k1         2         0       512  
//       x0        14      7168     28672  
//       x1        14       512      1024  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 512, 2, 2, 14, 14 }
// Out stride: { 1, 0, 0, 7168, 512 }
// Input 1 offset: 0
// Input 1 stride: { 1, 14336, 512, 28672, 1024 }
// Tile size: { 256, 1, 1, 14, 1 }
// Contraction output var shape: fp32(1, 14, 14, 512):(100352, 7168, 512, 1):392 KiB
// Computed true ops: 802816
// Computed work groups: 28
// Computed inner loops: 4
// Computed shared mem: 14392
// Computed out regs: 16384
// Computed mem read: 14336
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 2, 1
__kernel void kernel_c18_sdk_15(__global float* restrict  X_T80, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[16] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, };
  __local float in1_shared[3598];
  int c_gid = (get_group_id(1) * 256);
  int x1_gid = get_group_id(0);
  for (int k0_gid = 0; k0_gid < 2; k0_gid += 1)
  {
    for (int k1_gid = 0; k1_gid < 2; k1_gid += 1)
    {
      {
        int gbase = (((c_gid + (k1_gid * 512)) + (x1_gid * 1024)) + (k0_gid * 14336));
        int c_tid = (tid % 256);
        for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
        {
          int lidx = (c_tid + (257 * x0_lid));
          int gidx = ((gbase + c_tid) + (28672 * x0_lid));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)401407)];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      int c_tid = (tid % 32);
      int x0_tid = ((tid / 32) % 8);
      for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 1) || (x0_tid < 6));
        int x0 = select((int)0, (int)((8 * x0_lid) + x0_tid), (int)x0_cond);
        for (int c_lid = 0; c_lid < 8; c_lid += 1)
        {
          int c = ((32 * c_lid) + c_tid);
          float val1 = in1_shared[(c + (257 * x0))];
          int agg_idx = (c_lid + (x0_lid * 8));
          float agg_rhs = select((float)agg[agg_idx], (float)val1, (int)(val1 > agg[agg_idx]));
          agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)x0_cond);
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 32);
  int x0_tid = ((tid / 32) % 8);
  for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
  {
    int x0_cond = ((x0_lid < 1) || (x0_tid < 6));
    if (x0_cond)
    {
      int x0 = ((8 * x0_lid) + x0_tid);
      for (int c_lid = 0; c_lid < 8; c_lid += 1)
      {
        int c = ((32 * c_lid) + c_tid);
        float LX_T80 = agg[(c_lid + (x0_lid * 8))];
        LX_T80 = select((float)LX_T80, (float)0, (int)(LX_T80 == (float)-FLT_MAX));
        int gout_idx = (((c_gid + c) + (7168 * x0)) + (512 * x1_gid));
        X_T80[gout_idx] = LX_T80;
      }
    }
  }
}
