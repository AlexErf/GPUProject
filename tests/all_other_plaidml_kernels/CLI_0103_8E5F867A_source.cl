#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 9984 1 1
// lid: 256 1 1
// Original:
// X_T248[n, d0, d1, c : _T337, _T338, _T339, _T340] = =(X_T246[n, d0, d1, c])
// With Index Variables Made Integral:
// X_T248[n, d0, d1, c : _T337, _T338, _T339, _T340] = =(X_T246[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= c < 22, 0 <= c < 22, 0 <= d0 < 56, 0 <= d1 < 56, 0 <= d0 < 57, 0 <= d1 < 57, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= c < 22, 0 <= d0 < 56, 0 <= d1 < 56 }
// Defracted:
// X_T248[n, d0, d1, c : _T337, _T338, _T339, _T340] = =(X_T246[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range    X_T248    X_T246  
//        c        22         1         1  
//       d0        56      1254      1232  
//       d1        56        22        22  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 1232, 56 }
// Out stride: { 1, 1254 }
// Input 1 offset: 0
// Input 1 stride: { 1, 1232 }
// Tile size: { 32, 56 }
// Contraction output var shape: fp32(1, 57, 57, 22):(71478, 1254, 22, 1):279.211 KiB
// Computed true ops: 137984
// Computed work groups: 39
// Computed inner loops: 1
// Computed shared mem: 7296
// Computed out regs: 7168
// Computed mem read: 7168
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 9984, 1, 1
__kernel void kernel_c42_sdk_77(__global float* restrict  X_T248, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[7] = {0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[1824];
  int d1_c_gid = (get_group_id(0) * 32);
  {
    {
      int d1_c_tid = (tid % 32);
      int d0_tid = ((tid / 32) % 8);
      for (int d0_lid = 0; d0_lid < 7; d0_lid += 1)
      {
        int d0 = ((8 * d0_lid) + d0_tid);
        int lidx = ((57 * d1_c_tid) + d0);
        int gidx = ((d1_c_gid + d1_c_tid) + (1232 * d0));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)68991)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int d1_c_tid = (tid % 32);
    int d0_tid = ((tid / 32) % 8);
    for (int d0_lid = 0; d0_lid < 7; d0_lid += 1)
    {
      int d0 = ((8 * d0_lid) + d0_tid);
      float val1 = in1_shared[((57 * d1_c_tid) + d0)];
      agg[d0_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int d1_c_tid = (tid % 32);
  int d0_tid = ((tid / 32) % 8);
  int d1_c_cond = ((d1_c_gid != 1216) || (d1_c_tid < 16));
  if (d1_c_cond)
  {
    for (int d0_lid = 0; d0_lid < 7; d0_lid += 1)
    {
      int d0 = ((8 * d0_lid) + d0_tid);
      float LX_T248 = agg[d0_lid];
      int gout_idx = ((d1_c_gid + d1_c_tid) + (1254 * d0));
      X_T248[gout_idx] = LX_T248;
    }
  }
}
