#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2304 35 1
// lid: 256 1 1
// Original:
// X_T157[n0, n1, n2, 160 + a : _T167, _T168, _T169, _T170] = =(X_T156[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T157[n0, n1, n2, 160 + a : _T167, _T168, _T169, _T170] = =(X_T156[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 35, 0 <= n2 < 35, 0 <= n1 < 35, 0 <= n2 < 35, 0 <= a < 96, 0 <= 160 + a < 320, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 35, 0 <= n2 < 35, 0 <= a < 96 }
// Defracted:
// X_T157[n0, n1, n2, 160 + a : _T167, _T168, _T169, _T170] = =(X_T156[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T157    X_T156  
//        a        96         1         1  
//       n1        35     11200      3360  
//       n2        35       320        96  
//      off                 160         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 96, 35, 35 }
// Out stride: { 1, 11200, 320 }
// Input 1 offset: 0
// Input 1 stride: { 1, 3360, 96 }
// Tile size: { 96, 4, 1 }
// Contraction output var shape: fp32(1, 35, 35, 320):(392000, 11200, 320, 1):1531.25 KiB
// Computed true ops: 235200
// Computed work groups: 315
// Computed inner loops: 1
// Computed shared mem: 1552
// Computed out regs: 2048
// Computed mem read: 1536
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2304, 35, 1
__kernel void kernel_c51_sdk_38(__global float* restrict  X_T157, __global const float* restrict  in1)
{
  X_T157 = (X_T157 + 160);
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[388];
  int n2_gid = get_group_id(1);
  int n1_gid = (get_group_id(0) * 4);
  {
    {
      int gbase = ((n2_gid * 96) + (n1_gid * 3360));
      int a_n2_tid = (tid % 128);
      int n1_tid = ((tid / 128) % 2);
      int a_n2_cond = (a_n2_tid < 96);
      if (a_n2_cond)
      {
        for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
        {
          int n1 = ((2 * n1_lid) + n1_tid);
          int lidx = (a_n2_tid + (97 * n1));
          int gidx = ((gbase + a_n2_tid) + (3360 * n1));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)117599)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 64);
    int n1_tid = ((tid / 64) % 4);
    for (int a_lid = 0; a_lid < 2; a_lid += 1)
    {
      int a_cond = ((a_lid < 1) || (a_tid < 32));
      int a = select((int)0, (int)((64 * a_lid) + a_tid), (int)a_cond);
      float val1 = in1_shared[(a + (97 * n1_tid))];
      agg[a_lid] = select((float)agg[a_lid], (float)val1, (int)a_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 64);
  int n1_tid = ((tid / 64) % 4);
  int n1_cond = ((n1_gid != 32) || (n1_tid < 3));
  if (n1_cond)
  {
    for (int a_lid = 0; a_lid < 2; a_lid += 1)
    {
      int a_cond = ((a_lid < 1) || (a_tid < 32));
      if (a_cond)
      {
        int a = ((64 * a_lid) + a_tid);
        float LX_T157 = agg[a_lid];
        int gout_idx = ((a + (11200 * (n1_gid + n1_tid))) + (320 * n2_gid));
        X_T157[gout_idx] = LX_T157;
      }
    }
  }
}
