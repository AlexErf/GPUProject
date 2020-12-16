#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 14336 2 1
// lid: 256 1 1
// Original:
// X_T46[n, x0, x1, c : _T35, _T36, _T37, _T38] = >(X_T45[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2
// With Index Variables Made Integral:
// X_T46[n, x0, x1, c : _T35, _T36, _T37, _T38] = >(X_T45[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 2, 0 <= k1 < 2, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + 2*x0 < 112, 0 <= k1 + 2*x1 < 112, 0 <= c < 128, 0 <= c < 128, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 2, 0 <= k1 < 2, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + 2*x0 < 112, 0 <= k1 + 2*x1 < 112, 0 <= c < 128 }
// Defracted:
// X_T46[n, x0, x1, c : _T35, _T36, _T37, _T38] = >(X_T45[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T46     X_T45  
//        c       128         1         1  
//       k0         2         0     14336  
//       k1         2         0       128  
//       x0        56      7168     28672  
//       x1        56       128       256  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 128, 2, 2, 56, 56 }
// Out stride: { 1, 0, 0, 7168, 128 }
// Input 1 offset: 0
// Input 1 stride: { 1, 14336, 128, 28672, 256 }
// Tile size: { 64, 1, 1, 56, 1 }
// Contraction output var shape: fp32(1, 56, 56, 128):(401408, 7168, 128, 1):1568 KiB
// Computed true ops: 3211264
// Computed work groups: 112
// Computed inner loops: 4
// Computed shared mem: 14560
// Computed out regs: 14336
// Computed mem read: 14336
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 14336, 2, 1
__kernel void kernel_c18_sdk_5(__global float* restrict  X_T46, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[14] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, };
  __local float in1_shared[3640];
  int c_gid = (get_group_id(1) * 64);
  int x1_gid = get_group_id(0);
  for (int k0_gid = 0; k0_gid < 2; k0_gid += 1)
  {
    for (int k1_gid = 0; k1_gid < 2; k1_gid += 1)
    {
      {
        int gbase = (((c_gid + (k1_gid * 128)) + (x1_gid * 256)) + (k0_gid * 14336));
        int c_tid = (tid % 64);
        int x0_tid = ((tid / 64) % 4);
        for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
        {
          int x0 = ((4 * x0_lid) + x0_tid);
          int lidx = (c_tid + (65 * x0));
          int gidx = ((gbase + c_tid) + (28672 * x0));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)1605631)];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      int c_tid = (tid % 32);
      int x0_tid = ((tid / 32) % 8);
      for (int c_lid = 0; c_lid < 2; c_lid += 1)
      {
        int c = ((32 * c_lid) + c_tid);
        for (int x0_lid = 0; x0_lid < 7; x0_lid += 1)
        {
          int x0 = ((8 * x0_lid) + x0_tid);
          float val1 = in1_shared[(c + (65 * x0))];
          int agg_idx = (c_lid + (x0_lid * 2));
          float agg_rhs = select((float)agg[agg_idx], (float)val1, (int)(val1 > agg[agg_idx]));
          agg[agg_idx] = agg_rhs;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 32);
  int x0_tid = ((tid / 32) % 8);
  for (int c_lid = 0; c_lid < 2; c_lid += 1)
  {
    int c = ((32 * c_lid) + c_tid);
    for (int x0_lid = 0; x0_lid < 7; x0_lid += 1)
    {
      int x0 = ((8 * x0_lid) + x0_tid);
      float LX_T46 = agg[(c_lid + (x0_lid * 2))];
      LX_T46 = select((float)LX_T46, (float)0, (int)(LX_T46 == (float)-FLT_MAX));
      int gout_idx = (((c_gid + c) + (7168 * x0)) + (128 * x1_gid));
      X_T46[gout_idx] = LX_T46;
    }
  }
}
