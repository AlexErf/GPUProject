#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 8960 1 1
// lid: 256 1 1
// Original:
// X_T208[n0, n1, n2, 32 + a : _T252, _T253, _T254, _T255] = =(X_T207[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T208[n0, n1, n2, 32 + a : _T252, _T253, _T254, _T255] = =(X_T207[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= a < 32, 0 <= n1 < 35, 0 <= n2 < 35, 0 <= n1 < 35, 0 <= n2 < 35, 0 <= 32 + a < 128, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= a < 32, 0 <= n1 < 35, 0 <= n2 < 35 }
// Defracted:
// X_T208[n0, n1, n2, 32 + a : _T252, _T253, _T254, _T255] = =(X_T207[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T208    X_T207  
//        a        32         1         1  
//       n1        35      4480      1120  
//       n2        35       128        32  
//      off                  32         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 32, 35, 35 }
// Out stride: { 1, 4480, 128 }
// Input 1 offset: 0
// Input 1 stride: { 1, 1120, 32 }
// Tile size: { 32, 35, 1 }
// Contraction output var shape: fp32(1, 35, 35, 128):(156800, 4480, 128, 1):612.5 KiB
// Computed true ops: 78400
// Computed work groups: 35
// Computed inner loops: 1
// Computed shared mem: 4480
// Computed out regs: 5120
// Computed mem read: 4480
// Computed mem write: 4480
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 8960, 1, 1
__kernel void kernel_c51_sdk_58(__global float* restrict  X_T208, __global const float* restrict  in1)
{
  X_T208 = (X_T208 + 32);
  int tid = get_local_id(0);
  float agg[5] = {0, 0, 0, 0, 0, };
  __local float in1_shared[1120];
  int n2_gid = get_group_id(0);
  {
    {
      int gbase = (n2_gid * 32);
      int a_n2_tid = (tid % 32);
      int n1_tid = ((tid / 32) % 8);
      for (int n1_lid = 0; n1_lid < 5; n1_lid += 1)
      {
        int n1_cond = ((n1_lid < 4) || (n1_tid < 3));
        if (n1_cond)
        {
          int n1 = ((8 * n1_lid) + n1_tid);
          int lidx = ((35 * a_n2_tid) + n1);
          int gidx = ((gbase + a_n2_tid) + (1120 * n1));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)39199)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n1_tid = ((tid / 32) % 8);
    for (int n1_lid = 0; n1_lid < 5; n1_lid += 1)
    {
      int n1_cond = ((n1_lid < 4) || (n1_tid < 3));
      int n1 = select((int)0, (int)((8 * n1_lid) + n1_tid), (int)n1_cond);
      float val1 = in1_shared[((35 * a_tid) + n1)];
      agg[n1_lid] = select((float)agg[n1_lid], (float)val1, (int)n1_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n1_tid = ((tid / 32) % 8);
  for (int n1_lid = 0; n1_lid < 5; n1_lid += 1)
  {
    int n1_cond = ((n1_lid < 4) || (n1_tid < 3));
    if (n1_cond)
    {
      int n1 = ((8 * n1_lid) + n1_tid);
      float LX_T208 = agg[n1_lid];
      int gout_idx = ((a_tid + (4480 * n1)) + (128 * n2_gid));
      X_T208[gout_idx] = LX_T208;
    }
  }
}
