#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 8 1
// lid: 256 1 1
// Original:
// X_T2012[n0, n1, n2, a : _T2822, _T2823, _T2824, _T2825] = =(X_T2011[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T2012[n0, n1, n2, a : _T2822, _T2823, _T2824, _T2825] = =(X_T2011[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 8, 0 <= n2 < 8, 0 <= n1 < 8, 0 <= n2 < 8, 0 <= a < 192, 0 <= a < 448, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 8, 0 <= n2 < 8, 0 <= a < 192 }
// Defracted:
// X_T2012[n0, n1, n2, a : _T2822, _T2823, _T2824, _T2825] = =(X_T2011[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T2012   X_T2011  
//        a       192         1         1  
//       n1         8      3584      1536  
//       n2         8       448       192  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 192, 8, 8 }
// Out stride: { 1, 3584, 448 }
// Input 1 offset: 0
// Input 1 stride: { 1, 1536, 192 }
// Tile size: { 192, 4, 1 }
// Contraction output var shape: fp32(1, 8, 8, 448):(28672, 3584, 448, 1):112 KiB
// Computed true ops: 24576
// Computed work groups: 16
// Computed inner loops: 1
// Computed shared mem: 3088
// Computed out regs: 3072
// Computed mem read: 3072
// Computed mem write: 3072
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 512, 8, 1
__kernel void kernel_c51_sdk_658(__global float* restrict  X_T2012, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[3] = {0, 0, 0, };
  __local float in1_shared[772];
  int n2_gid = get_group_id(1);
  int n1_gid = (get_group_id(0) * 4);
  {
    {
      int gbase = ((n2_gid * 192) + (n1_gid * 1536));
      int a_n2_tid = (tid % 256);
      int a_n2_cond = (a_n2_tid < 192);
      if (a_n2_cond)
      {
        for (int n1_lid = 0; n1_lid < 4; n1_lid += 1)
        {
          int lidx = (a_n2_tid + (193 * n1_lid));
          int gidx = ((gbase + a_n2_tid) + (1536 * n1_lid));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)12287)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 64);
    int n1_tid = ((tid / 64) % 4);
    for (int a_lid = 0; a_lid < 3; a_lid += 1)
    {
      int a = ((64 * a_lid) + a_tid);
      float val1 = in1_shared[(a + (193 * n1_tid))];
      agg[a_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 64);
  int n1_tid = ((tid / 64) % 4);
  for (int a_lid = 0; a_lid < 3; a_lid += 1)
  {
    int a = ((64 * a_lid) + a_tid);
    float LX_T2012 = agg[a_lid];
    int gout_idx = ((a + (3584 * (n1_gid + n1_tid))) + (448 * n2_gid));
    X_T2012[gout_idx] = LX_T2012;
  }
}
