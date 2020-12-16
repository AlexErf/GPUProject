#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 8 1
// lid: 256 1 1
// Original:
// X_T943[n0, n1, n2, 384 + a : _T1326, _T1327, _T1328, _T1329] = =(X_T942[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T943[n0, n1, n2, 384 + a : _T1326, _T1327, _T1328, _T1329] = =(X_T942[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 8, 0 <= n2 < 8, 0 <= n1 < 8, 0 <= n2 < 8, 0 <= a < 384, 0 <= 384 + a < 768, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 8, 0 <= n2 < 8, 0 <= a < 384 }
// Defracted:
// X_T943[n0, n1, n2, 384 + a : _T1326, _T1327, _T1328, _T1329] = =(X_T942[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T943    X_T942  
//        a       384         1         1  
//       n1         8      6144      3072  
//       n2         8       768       384  
//      off                 384         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 384, 8, 8 }
// Out stride: { 1, 6144, 768 }
// Input 1 offset: 0
// Input 1 stride: { 1, 3072, 384 }
// Tile size: { 384, 4, 1 }
// Contraction output var shape: fp32(1, 8, 8, 768):(49152, 6144, 768, 1):192 KiB
// Computed true ops: 49152
// Computed work groups: 16
// Computed inner loops: 1
// Computed shared mem: 6160
// Computed out regs: 6144
// Computed mem read: 6144
// Computed mem write: 6144
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 512, 8, 1
__kernel void kernel_c56_sdk_320(__global float* restrict  X_T943, __global const float* restrict  in1)
{
  X_T943 = (X_T943 + 384);
  int tid = get_local_id(0);
  float agg[6] = {0, 0, 0, 0, 0, 0, };
  __local float in1_shared[1540];
  int n2_gid = get_group_id(1);
  int n1_gid = (get_group_id(0) * 4);
  {
    {
      int gbase = ((n2_gid * 384) + (n1_gid * 3072));
      int a_n2_tid = (tid % 256);
      for (int a_n2_lid = 0; a_n2_lid < 2; a_n2_lid += 1)
      {
        int a_n2_cond = ((a_n2_lid < 1) || (a_n2_tid < 128));
        if (a_n2_cond)
        {
          int a_n2 = ((256 * a_n2_lid) + a_n2_tid);
          for (int n1_lid = 0; n1_lid < 4; n1_lid += 1)
          {
            int lidx = (a_n2 + (385 * n1_lid));
            int gidx = ((gbase + a_n2) + (3072 * n1_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)24575)];
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 64);
    int n1_tid = ((tid / 64) % 4);
    for (int a_lid = 0; a_lid < 6; a_lid += 1)
    {
      int a = ((64 * a_lid) + a_tid);
      float val1 = in1_shared[(a + (385 * n1_tid))];
      agg[a_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 64);
  int n1_tid = ((tid / 64) % 4);
  for (int a_lid = 0; a_lid < 6; a_lid += 1)
  {
    int a = ((64 * a_lid) + a_tid);
    float LX_T943 = agg[a_lid];
    int gout_idx = ((a + (6144 * (n1_gid + n1_tid))) + (768 * n2_gid));
    X_T943[gout_idx] = LX_T943;
  }
}
