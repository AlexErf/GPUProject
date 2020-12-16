#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Original:
// X_T1028[n0, n1, n2, a : _T1463, _T1464, _T1465, _T1466] = =(X_T1027[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T1028[n0, n1, n2, a : _T1463, _T1464, _T1465, _T1466] = =(X_T1027[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 864, 0 <= a < 896, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 864 }
// Defracted:
// X_T1028[n0, n1, n2, a : _T1463, _T1464, _T1465, _T1466] = =(X_T1027[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T1028   X_T1027  
//        a       864         1         1  
//       n1        14     12544     12096  
//       n2        14       896       864  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 864, 14, 14 }
// Out stride: { 1, 12544, 896 }
// Input 1 offset: 0
// Input 1 stride: { 1, 12096, 864 }
// Tile size: { 864, 2, 2 }
// Contraction output var shape: fp32(1, 14, 14, 896):(175616, 12544, 896, 1):686 KiB
// Computed true ops: 338688
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 13832
// Computed out regs: 14336
// Computed mem read: 13824
// Computed mem write: 13824
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c68_sdk_351(__global float* restrict  X_T1028, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3458];
  int n2_gid = (get_group_id(0) * 2);
  int n1_gid = (get_group_id(1) * 2);
  {
    {
      int gbase = ((n2_gid * 864) + (n1_gid * 12096));
      int a_n2_tid = (tid % 256);
      for (int a_n2_lid = 0; a_n2_lid < 7; a_n2_lid += 1)
      {
        int a_n2_cond = ((a_n2_lid < 6) || (a_n2_tid < 192));
        if (a_n2_cond)
        {
          int a_n2 = ((256 * a_n2_lid) + a_n2_tid);
          for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
          {
            int lidx = (a_n2 + (1729 * n1_lid));
            int gidx = ((gbase + a_n2) + (12096 * n1_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)169343)];
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 64);
    int n2_tid = ((tid / 64) % 2);
    int n1_tid = ((tid / 128) % 2);
    for (int a_lid = 0; a_lid < 14; a_lid += 1)
    {
      int a_cond = ((a_lid < 13) || (a_tid < 32));
      int a = select((int)0, (int)((64 * a_lid) + a_tid), (int)a_cond);
      float val1 = in1_shared[((a + (864 * n2_tid)) + (1729 * n1_tid))];
      agg[a_lid] = select((float)agg[a_lid], (float)val1, (int)a_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 64);
  int n2_tid = ((tid / 64) % 2);
  int n1_tid = ((tid / 128) % 2);
  for (int a_lid = 0; a_lid < 14; a_lid += 1)
  {
    int a_cond = ((a_lid < 13) || (a_tid < 32));
    if (a_cond)
    {
      int a = ((64 * a_lid) + a_tid);
      float LX_T1028 = agg[a_lid];
      int gout_idx = ((a + (12544 * (n1_gid + n1_tid))) + (896 * (n2_gid + n2_tid)));
      X_T1028[gout_idx] = LX_T1028;
    }
  }
}
