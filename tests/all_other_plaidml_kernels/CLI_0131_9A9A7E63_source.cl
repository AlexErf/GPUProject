#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 4352 17 1
// lid: 256 1 1
// Original:
// X_T897[n0, n1, n2, a : _T1232, _T1233, _T1234, _T1235] = =(X_T896[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T897[n0, n1, n2, a : _T1232, _T1233, _T1234, _T1235] = =(X_T896[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 17, 0 <= n2 < 17, 0 <= n1 < 17, 0 <= n2 < 17, 0 <= a < 384, 0 <= a < 1088, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 17, 0 <= n2 < 17, 0 <= a < 384 }
// Defracted:
// X_T897[n0, n1, n2, a : _T1232, _T1233, _T1234, _T1235] = =(X_T896[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T897    X_T896  
//        a       384         1         1  
//       n1        17     18496      6528  
//       n2        17      1088       384  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 384, 17, 17 }
// Out stride: { 1, 18496, 1088 }
// Input 1 offset: 0
// Input 1 stride: { 1, 6528, 384 }
// Tile size: { 384, 1, 1 }
// Contraction output var shape: fp32(1, 17, 17, 1088):(314432, 18496, 1088, 1):1228.25 KiB
// Computed true ops: 221952
// Computed work groups: 289
// Computed inner loops: 1
// Computed shared mem: 1536
// Computed out regs: 2048
// Computed mem read: 1536
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 4352, 17, 1
__kernel void kernel_c51_sdk_291(__global float* restrict  X_T897, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[384];
  int n2_gid = get_group_id(0);
  int n1_gid = get_group_id(1);
  {
    {
      int gbase = ((n2_gid * 384) + (n1_gid * 6528));
      int a_n2_tid = (tid % 256);
      for (int a_n2_lid = 0; a_n2_lid < 2; a_n2_lid += 1)
      {
        int a_n2_cond = ((a_n2_lid < 1) || (a_n2_tid < 128));
        if (a_n2_cond)
        {
          int a_n2 = ((256 * a_n2_lid) + a_n2_tid);
          int gidx = (gbase + a_n2);
          in1_shared[a_n2] = in1[clamp((int)gidx, (int)0, (int)110975)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 256);
    for (int a_lid = 0; a_lid < 2; a_lid += 1)
    {
      int a_cond = ((a_lid < 1) || (a_tid < 128));
      int a = select((int)0, (int)((256 * a_lid) + a_tid), (int)a_cond);
      float val1 = in1_shared[a];
      agg[a_lid] = select((float)agg[a_lid], (float)val1, (int)a_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 256);
  for (int a_lid = 0; a_lid < 2; a_lid += 1)
  {
    int a_cond = ((a_lid < 1) || (a_tid < 128));
    if (a_cond)
    {
      int a = ((256 * a_lid) + a_tid);
      float LX_T897 = agg[a_lid];
      int gout_idx = ((a + (18496 * n1_gid)) + (1088 * n2_gid));
      X_T897[gout_idx] = LX_T897;
    }
  }
}
