#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7424 1 1
// lid: 256 1 1
// Original:
// X_T317[n, o0, o1, c : _T473, _T474, _T475, _T476] = =(X_T97[])
// With Index Variables Made Integral:
// X_T317[n, o0, o1, c : _T473, _T474, _T475, _T476] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= c < 22, 0 <= o0 < 57, 0 <= o1 < 57, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= c < 22, 0 <= o0 < 57, 0 <= o1 < 57 }
// Defracted:
// X_T317[n, o0, o1, c : _T473, _T474, _T475, _T476] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range    X_T317     X_T97  
//        c        22         1         0  
//       o0        57      1254         0  
//       o1        57        22         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 22, 57, 57 }
// Out stride: { 1, 1254, 22 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 22, 2, 57 }
// Contraction output var shape: fp32(1, 57, 57, 22):(71478, 1254, 22, 1):279.211 KiB
// Computed true ops: 142956
// Computed work groups: 29
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 15360
// Computed mem read: 128
// Computed mem write: 14592
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7424, 1, 1
__kernel void kernel_c42_sdk_107(__global float* restrict  X_T317)
{
  int tid = get_local_id(0);
  float agg[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  int o0_gid = (get_group_id(0) * 2);
  {
    int c_tid = (tid % 32);
    int o1_tid = ((tid / 32) % 4);
    int o0_tid = ((tid / 128) % 2);
    int c_cond = (c_tid < 22);
    int c = select((int)0, (int)c_tid, (int)c_cond);
    for (int o1_lid = 0; o1_lid < 15; o1_lid += 1)
    {
      int o1_cond = ((o1_lid < 14) || (o1_tid < 1));
      int o1 = select((int)0, (int)((4 * o1_lid) + o1_tid), (int)(c_cond && o1_cond));
      float val1 = 1.0f;
      agg[o1_lid] = select((float)agg[o1_lid], (float)val1, (int)(c_cond && o1_cond));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o1_tid = ((tid / 32) % 4);
  int o0_tid = ((tid / 128) % 2);
  int o0_cond = ((o0_gid != 56) || (o0_tid < 1));
  if (o0_cond)
  {
    int c_cond = (c_tid < 22);
    if (c_cond)
    {
      for (int o1_lid = 0; o1_lid < 15; o1_lid += 1)
      {
        int o1_cond = ((o1_lid < 14) || (o1_tid < 1));
        if (o1_cond)
        {
          int o1 = ((4 * o1_lid) + o1_tid);
          float LX_T317 = agg[o1_lid];
          int gout_idx = ((c_tid + (1254 * (o0_gid + o0_tid))) + (22 * o1));
          X_T317[gout_idx] = LX_T317;
        }
      }
    }
  }
}
