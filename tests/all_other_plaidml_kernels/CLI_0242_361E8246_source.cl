#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 14 1
// lid: 256 1 1
// Original:
// X_T403[n0, n1, n2, a : _T507, _T508, _T509, _T510] = =(X_T402[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T403[n0, n1, n2, a : _T507, _T508, _T509, _T510] = =(X_T402[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= a < 320, 0 <= a < 352, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= a < 320 }
// Defracted:
// X_T403[n0, n1, n2, a : _T507, _T508, _T509, _T510] = =(X_T402[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T403    X_T402  
//        a       320         1         1  
//       n1        28      9856      8960  
//       n2        28       352       320  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 320, 28, 28 }
// Out stride: { 1, 9856, 352 }
// Input 1 offset: 0
// Input 1 stride: { 1, 8960, 320 }
// Tile size: { 320, 2, 2 }
// Contraction output var shape: fp32(1, 28, 28, 352):(275968, 9856, 352, 1):1078 KiB
// Computed true ops: 501760
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 5128
// Computed out regs: 5120
// Computed mem read: 5120
// Computed mem write: 5120
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 14, 1
__kernel void kernel_c108_sdk_120(__global float* restrict  X_T403, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[5] = {0, 0, 0, 0, 0, };
  __local float in1_shared[1282];
  int n2_gid = (get_group_id(0) * 2);
  int n1_gid = (get_group_id(1) * 2);
  {
    {
      int gbase = ((n2_gid * 320) + (n1_gid * 8960));
      int a_n2_tid = (tid % 256);
      for (int a_n2_lid = 0; a_n2_lid < 3; a_n2_lid += 1)
      {
        int a_n2_cond = ((a_n2_lid < 2) || (a_n2_tid < 128));
        if (a_n2_cond)
        {
          int a_n2 = ((256 * a_n2_lid) + a_n2_tid);
          for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
          {
            int lidx = (a_n2 + (641 * n1_lid));
            int gidx = ((gbase + a_n2) + (8960 * n1_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)250879)];
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 64);
    int n2_tid = ((tid / 64) % 2);
    int n1_tid = ((tid / 128) % 2);
    for (int a_lid = 0; a_lid < 5; a_lid += 1)
    {
      int a = ((64 * a_lid) + a_tid);
      float val1 = in1_shared[((a + (320 * n2_tid)) + (641 * n1_tid))];
      agg[a_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 64);
  int n2_tid = ((tid / 64) % 2);
  int n1_tid = ((tid / 128) % 2);
  for (int a_lid = 0; a_lid < 5; a_lid += 1)
  {
    int a = ((64 * a_lid) + a_tid);
    float LX_T403 = agg[a_lid];
    int gout_idx = ((a + (9856 * (n1_gid + n1_tid))) + (352 * (n2_gid + n2_tid)));
    X_T403[gout_idx] = LX_T403;
  }
}
