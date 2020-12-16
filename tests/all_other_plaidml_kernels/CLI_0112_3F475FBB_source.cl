#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 83 1
// lid: 256 1 1
// Original:
// X_T278[n0, n1, n2, 42 + a : _T405, _T406, _T407, _T408] = =(X_T277[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T278[n0, n1, n2, 42 + a : _T405, _T406, _T407, _T408] = =(X_T277[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= a < 42, 0 <= n1 < 83, 0 <= n2 < 83, 0 <= n1 < 83, 0 <= n2 < 83, 0 <= 42 + a < 84, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= a < 42, 0 <= n1 < 83, 0 <= n2 < 83 }
// Defracted:
// X_T278[n0, n1, n2, 42 + a : _T405, _T406, _T407, _T408] = =(X_T277[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T278    X_T277  
//        a        42         1         1  
//       n1        83      6972      3486  
//       n2        83        84        42  
//      off                  42         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 42, 83, 83 }
// Out stride: { 1, 6972, 84 }
// Input 1 offset: 0
// Input 1 stride: { 1, 3486, 42 }
// Tile size: { 42, 1, 4 }
// Contraction output var shape: fp32(1, 83, 83, 84):(578676, 6972, 84, 1):2260.45 KiB
// Computed true ops: 578676
// Computed work groups: 1743
// Computed inner loops: 1
// Computed shared mem: 672
// Computed out regs: 1024
// Computed mem read: 640
// Computed mem write: 1024
// Computed operations: 168
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 83, 1
__kernel void kernel_c42_sdk_92(__global float* restrict  X_T278, __global const float* restrict  in1)
{
  X_T278 = (X_T278 + 42);
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[168];
  int n2_gid = (get_group_id(0) * 4);
  int n1_gid = get_group_id(1);
  {
    {
      int gbase = ((n2_gid * 42) + (n1_gid * 3486));
      int a_n2_tid = (tid % 256);
      int a_n2_cond = (a_n2_tid < 168);
      if (a_n2_cond)
      {
        int gidx = (gbase + a_n2_tid);
        in1_shared[a_n2_tid] = in1[clamp((int)gidx, (int)0, (int)289337)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 64);
    int n2_tid = ((tid / 64) % 4);
    int a_cond = (a_tid < 42);
    int a = select((int)0, (int)a_tid, (int)a_cond);
    float val1 = in1_shared[(a + (42 * n2_tid))];
    agg[0] = select((float)agg[0], (float)val1, (int)a_cond);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 64);
  int n2_tid = ((tid / 64) % 4);
  int n2_cond = ((n2_gid != 80) || (n2_tid < 3));
  if (n2_cond)
  {
    int a_cond = (a_tid < 42);
    if (a_cond)
    {
      float LX_T278 = agg[0];
      int gout_idx = ((a_tid + (6972 * n1_gid)) + (84 * (n2_gid + n2_tid)));
      X_T278[gout_idx] = LX_T278;
    }
  }
}
