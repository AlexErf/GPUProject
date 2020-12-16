#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 14 1
// lid: 256 1 1
// Original:
// X_T500[n0, n1, n2, 22 + a : _T768, _T769, _T770, _T771] = =(X_T499[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T500[n0, n1, n2, 22 + a : _T768, _T769, _T770, _T771] = =(X_T499[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= a < 22, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= 22 + a < 44, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= a < 22, 0 <= n1 < 28, 0 <= n2 < 28 }
// Defracted:
// X_T500[n0, n1, n2, 22 + a : _T768, _T769, _T770, _T771] = =(X_T499[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T500    X_T499  
//        a        22         1         1  
//       n1        28      1232       616  
//       n2        28        44        22  
//      off                  22         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 22, 28, 28 }
// Out stride: { 1, 1232, 44 }
// Input 1 offset: 0
// Input 1 stride: { 1, 616, 22 }
// Tile size: { 22, 2, 4 }
// Contraction output var shape: fp32(1, 28, 28, 44):(34496, 1232, 44, 1):134.75 KiB
// Computed true ops: 34496
// Computed work groups: 98
// Computed inner loops: 1
// Computed shared mem: 712
// Computed out regs: 1024
// Computed mem read: 640
// Computed mem write: 1024
// Computed operations: 176
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 14, 1
__kernel void kernel_c42_sdk_177(__global float* restrict  X_T500, __global const float* restrict  in1)
{
  X_T500 = (X_T500 + 22);
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[178];
  int n2_gid = (get_group_id(0) * 4);
  int n1_gid = (get_group_id(1) * 2);
  {
    {
      int gbase = ((n2_gid * 22) + (n1_gid * 616));
      int a_n2_tid = (tid % 128);
      int n1_tid = ((tid / 128) % 2);
      int a_n2_cond = (a_n2_tid < 88);
      if (a_n2_cond)
      {
        int lidx = (a_n2_tid + (89 * n1_tid));
        int gidx = ((gbase + a_n2_tid) + (616 * n1_tid));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)17247)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 4);
    int n1_tid = ((tid / 128) % 2);
    int a_cond = (a_tid < 22);
    int a = select((int)0, (int)a_tid, (int)a_cond);
    float val1 = in1_shared[((a + (22 * n2_tid)) + (89 * n1_tid))];
    agg[0] = select((float)agg[0], (float)val1, (int)a_cond);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 4);
  int n1_tid = ((tid / 128) % 2);
  int a_cond = (a_tid < 22);
  if (a_cond)
  {
    float LX_T500 = agg[0];
    int gout_idx = ((a_tid + (1232 * (n1_gid + n1_tid))) + (44 * (n2_gid + n2_tid)));
    X_T500[gout_idx] = LX_T500;
  }
}
