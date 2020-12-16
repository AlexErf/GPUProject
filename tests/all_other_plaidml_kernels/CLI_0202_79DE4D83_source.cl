#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Original:
// X_T230[n0, n1, n2, 224 + a : _T248, _T249, _T250, _T251] = =(X_T229[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T230[n0, n1, n2, 224 + a : _T248, _T249, _T250, _T251] = =(X_T229[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= a < 32, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= 224 + a < 256, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= a < 32, 0 <= n1 < 56, 0 <= n2 < 56 }
// Defracted:
// X_T230[n0, n1, n2, 224 + a : _T248, _T249, _T250, _T251] = =(X_T229[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T230    X_T229  
//        a        32         1         1  
//       n1        56     14336      1792  
//       n2        56       256        32  
//      off                 224         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 32, 56, 56 }
// Out stride: { 1, 14336, 256 }
// Input 1 offset: 0
// Input 1 stride: { 1, 1792, 32 }
// Tile size: { 32, 56, 2 }
// Contraction output var shape: fp32(1, 56, 56, 256):(802816, 14336, 256, 1):3136 KiB
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
__kernel void kernel_c108_sdk_58(__global float* restrict  X_T230, __global const float* restrict  in1)
{
  X_T230 = (X_T230 + 224);
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
    float LX_T230 = agg[n1_lid];
    int gout_idx = ((a_tid + (14336 * n1)) + (256 * (n2_gid + n2_tid)));
    X_T230[gout_idx] = LX_T230;
  }
}
