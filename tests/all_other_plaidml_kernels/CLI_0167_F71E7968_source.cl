#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 14 1
// lid: 256 1 1
// Original:
// X_T658[n0, n1, n2, 176 + a : _T1014, _T1015, _T1016, _T1017] = =(X_T657[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T658[n0, n1, n2, 176 + a : _T1014, _T1015, _T1016, _T1017] = =(X_T657[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= a < 44, 0 <= 176 + a < 264, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= a < 44 }
// Defracted:
// X_T658[n0, n1, n2, 176 + a : _T1014, _T1015, _T1016, _T1017] = =(X_T657[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T658    X_T657  
//        a        44         1         1  
//       n1        28      7392      1232  
//       n2        28       264        44  
//      off                 176         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 44, 28, 28 }
// Out stride: { 1, 7392, 264 }
// Input 1 offset: 0
// Input 1 stride: { 1, 1232, 44 }
// Tile size: { 44, 2, 2 }
// Contraction output var shape: fp32(1, 28, 28, 264):(206976, 7392, 264, 1):808.5 KiB
// Computed true ops: 68992
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 712
// Computed out regs: 1024
// Computed mem read: 640
// Computed mem write: 1024
// Computed operations: 176
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 14, 1
__kernel void kernel_c42_sdk_237(__global float* restrict  X_T658, __global const float* restrict  in1)
{
  X_T658 = (X_T658 + 176);
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[178];
  int n2_gid = (get_group_id(0) * 2);
  int n1_gid = (get_group_id(1) * 2);
  {
    {
      int gbase = ((n2_gid * 44) + (n1_gid * 1232));
      int a_n2_tid = (tid % 128);
      int n1_tid = ((tid / 128) % 2);
      int a_n2_cond = (a_n2_tid < 88);
      if (a_n2_cond)
      {
        int lidx = (a_n2_tid + (89 * n1_tid));
        int gidx = ((gbase + a_n2_tid) + (1232 * n1_tid));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)34495)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 64);
    int n2_tid = ((tid / 64) % 2);
    int n1_tid = ((tid / 128) % 2);
    int a_cond = (a_tid < 44);
    int a = select((int)0, (int)a_tid, (int)a_cond);
    float val1 = in1_shared[((a + (44 * n2_tid)) + (89 * n1_tid))];
    agg[0] = select((float)agg[0], (float)val1, (int)a_cond);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 64);
  int n2_tid = ((tid / 64) % 2);
  int n1_tid = ((tid / 128) % 2);
  int a_cond = (a_tid < 44);
  if (a_cond)
  {
    float LX_T658 = agg[0];
    int gout_idx = ((a_tid + (7392 * (n1_gid + n1_tid))) + (264 * (n2_gid + n2_tid)));
    X_T658[gout_idx] = LX_T658;
  }
}
