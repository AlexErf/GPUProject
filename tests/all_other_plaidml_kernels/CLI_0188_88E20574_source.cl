#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 8 1
// lid: 256 1 1
// Original:
// X_T851[n0, n1, n2, a : _T1194, _T1195, _T1196, _T1197] = =(X_T850[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T851[n0, n1, n2, a : _T1194, _T1195, _T1196, _T1197] = =(X_T850[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 8, 0 <= n2 < 8, 0 <= n1 < 8, 0 <= n2 < 8, 0 <= a < 320, 0 <= a < 1280, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 8, 0 <= n2 < 8, 0 <= a < 320 }
// Defracted:
// X_T851[n0, n1, n2, a : _T1194, _T1195, _T1196, _T1197] = =(X_T850[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T851    X_T850  
//        a       320         1         1  
//       n1         8     10240      2560  
//       n2         8      1280       320  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 320, 8, 8 }
// Out stride: { 1, 10240, 1280 }
// Input 1 offset: 0
// Input 1 stride: { 1, 2560, 320 }
// Tile size: { 320, 4, 1 }
// Contraction output var shape: fp32(1, 8, 8, 1280):(81920, 10240, 1280, 1):320 KiB
// Computed true ops: 40960
// Computed work groups: 16
// Computed inner loops: 1
// Computed shared mem: 5136
// Computed out regs: 5120
// Computed mem read: 5120
// Computed mem write: 5120
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 512, 8, 1
__kernel void kernel_c56_sdk_288(__global float* restrict  X_T851, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[5] = {0, 0, 0, 0, 0, };
  __local float in1_shared[1284];
  int n2_gid = get_group_id(1);
  int n1_gid = (get_group_id(0) * 4);
  {
    {
      int gbase = ((n2_gid * 320) + (n1_gid * 2560));
      int a_n2_tid = (tid % 256);
      for (int a_n2_lid = 0; a_n2_lid < 2; a_n2_lid += 1)
      {
        int a_n2_cond = ((a_n2_lid < 1) || (a_n2_tid < 64));
        if (a_n2_cond)
        {
          int a_n2 = ((256 * a_n2_lid) + a_n2_tid);
          for (int n1_lid = 0; n1_lid < 4; n1_lid += 1)
          {
            int lidx = (a_n2 + (321 * n1_lid));
            int gidx = ((gbase + a_n2) + (2560 * n1_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)20479)];
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 64);
    int n1_tid = ((tid / 64) % 4);
    for (int a_lid = 0; a_lid < 5; a_lid += 1)
    {
      int a = ((64 * a_lid) + a_tid);
      float val1 = in1_shared[(a + (321 * n1_tid))];
      agg[a_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 64);
  int n1_tid = ((tid / 64) % 4);
  for (int a_lid = 0; a_lid < 5; a_lid += 1)
  {
    int a = ((64 * a_lid) + a_tid);
    float LX_T851 = agg[a_lid];
    int gout_idx = ((a + (10240 * (n1_gid + n1_tid))) + (1280 * n2_gid));
    X_T851[gout_idx] = LX_T851;
  }
}
