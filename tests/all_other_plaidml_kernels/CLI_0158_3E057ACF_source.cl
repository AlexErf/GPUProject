#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2304 9 1
// lid: 256 1 1
// Original:
// X_T382[n0, n1, n2, 480 + a : _T529, _T530, _T531, _T532] = =(X_T381[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T382[n0, n1, n2, 480 + a : _T529, _T530, _T531, _T532] = =(X_T381[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 17, 0 <= n2 < 17, 0 <= n1 < 17, 0 <= n2 < 17, 0 <= a < 288, 0 <= 480 + a < 768, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 17, 0 <= n2 < 17, 0 <= a < 288 }
// Defracted:
// X_T382[n0, n1, n2, 480 + a : _T529, _T530, _T531, _T532] = =(X_T381[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T382    X_T381  
//        a       288         1         1  
//       n1        17     13056      4896  
//       n2        17       768       288  
//      off                 480         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 288, 17, 17 }
// Out stride: { 1, 13056, 768 }
// Input 1 offset: 0
// Input 1 stride: { 1, 4896, 288 }
// Tile size: { 32, 2, 17 }
// Contraction output var shape: fp32(1, 17, 17, 768):(221952, 13056, 768, 1):867 KiB
// Computed true ops: 166464
// Computed work groups: 81
// Computed inner loops: 1
// Computed shared mem: 4480
// Computed out regs: 5120
// Computed mem read: 4352
// Computed mem write: 4352
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2304, 9, 1
__kernel void kernel_c56_sdk_126(__global float* restrict  X_T382, __global const float* restrict  in1)
{
  X_T382 = (X_T382 + 480);
  int tid = get_local_id(0);
  float agg[5] = {0, 0, 0, 0, 0, };
  __local float in1_shared[1120];
  int a_gid = (get_group_id(0) * 32);
  int n1_gid = (get_group_id(1) * 2);
  {
    {
      int gbase = (a_gid + (n1_gid * 4896));
      int a_tid = (tid % 32);
      int n2_n1_tid = ((tid / 32) % 8);
      for (int n2_n1_lid = 0; n2_n1_lid < 5; n2_n1_lid += 1)
      {
        int n2_n1_cond = ((n2_n1_lid < 4) || (n2_n1_tid < 2));
        if (n2_n1_cond)
        {
          int n2_n1 = ((8 * n2_n1_lid) + n2_n1_tid);
          int lidx = ((35 * a_tid) + n2_n1);
          int gidx = ((gbase + a_tid) + (288 * n2_n1));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)83231)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 4);
    int n1_tid = ((tid / 128) % 2);
    for (int n2_lid = 0; n2_lid < 5; n2_lid += 1)
    {
      int n2_cond = ((n2_lid < 4) || (n2_tid < 1));
      int n2 = select((int)0, (int)((4 * n2_lid) + n2_tid), (int)n2_cond);
      float val1 = in1_shared[(((35 * a_tid) + n2) + (17 * n1_tid))];
      agg[n2_lid] = select((float)agg[n2_lid], (float)val1, (int)n2_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 4);
  int n1_tid = ((tid / 128) % 2);
  int n1_cond = ((n1_gid != 16) || (n1_tid < 1));
  if (n1_cond)
  {
    for (int n2_lid = 0; n2_lid < 5; n2_lid += 1)
    {
      int n2_cond = ((n2_lid < 4) || (n2_tid < 1));
      if (n2_cond)
      {
        int n2 = ((4 * n2_lid) + n2_tid);
        float LX_T382 = agg[n2_lid];
        int gout_idx = (((a_gid + a_tid) + (13056 * (n1_gid + n1_tid))) + (768 * n2));
        X_T382[gout_idx] = LX_T382;
      }
    }
  }
}
