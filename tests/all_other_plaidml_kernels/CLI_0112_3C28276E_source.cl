#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 28 1
// lid: 256 1 1
// Original:
// X_T276[n0, n1, n2, 11 + a : _T405, _T406, _T407, _T408] = =(X_T275[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T276[n0, n1, n2, 11 + a : _T405, _T406, _T407, _T408] = =(X_T275[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= a < 11, 0 <= 11 + a < 22, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= a < 11, 0 <= n1 < 56, 0 <= n2 < 56 }
// Defracted:
// X_T276[n0, n1, n2, 11 + a : _T405, _T406, _T407, _T408] = =(X_T275[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T276    X_T275  
//        a        11         1         1  
//       n1        56      1232       616  
//       n2        56        22        11  
//      off                  11         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 11, 56, 56 }
// Out stride: { 1, 1232, 22 }
// Input 1 offset: 0
// Input 1 stride: { 1, 616, 11 }
// Tile size: { 11, 2, 8 }
// Contraction output var shape: fp32(1, 56, 56, 22):(68992, 1232, 22, 1):269.5 KiB
// Computed true ops: 68992
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 712
// Computed out regs: 1024
// Computed mem read: 640
// Computed mem write: 2048
// Computed operations: 176
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 28, 1
__kernel void kernel_c42_sdk_92(__global float* restrict  X_T276, __global const float* restrict  in1)
{
  X_T276 = (X_T276 + 11);
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[178];
  int n2_gid = (get_group_id(0) * 8);
  int n1_gid = (get_group_id(1) * 2);
  {
    {
      int gbase = ((n2_gid * 11) + (n1_gid * 616));
      int a_n2_tid = (tid % 128);
      int n1_tid = ((tid / 128) % 2);
      int a_n2_cond = (a_n2_tid < 88);
      if (a_n2_cond)
      {
        int lidx = (a_n2_tid + (89 * n1_tid));
        int gidx = ((gbase + a_n2_tid) + (616 * n1_tid));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)34495)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 16);
    int n2_tid = ((tid / 16) % 8);
    int n1_tid = ((tid / 128) % 2);
    int a_cond = (a_tid < 11);
    int a = select((int)0, (int)a_tid, (int)a_cond);
    float val1 = in1_shared[((a + (11 * n2_tid)) + (89 * n1_tid))];
    agg[0] = select((float)agg[0], (float)val1, (int)a_cond);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 16);
  int n2_tid = ((tid / 16) % 8);
  int n1_tid = ((tid / 128) % 2);
  int a_cond = (a_tid < 11);
  if (a_cond)
  {
    float LX_T276 = agg[0];
    int gout_idx = ((a_tid + (1232 * (n1_gid + n1_tid))) + (22 * (n2_gid + n2_tid)));
    X_T276[gout_idx] = LX_T276;
  }
}
