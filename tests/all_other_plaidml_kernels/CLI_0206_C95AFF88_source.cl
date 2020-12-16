#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 14 1
// lid: 256 1 1
// Original:
// X_T483[n0, n1, n2, a : _T655, _T656, _T657, _T658] = =(X_T482[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T483[n0, n1, n2, a : _T655, _T656, _T657, _T658] = =(X_T482[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= a < 448, 0 <= a < 480, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= a < 448 }
// Defracted:
// X_T483[n0, n1, n2, a : _T655, _T656, _T657, _T658] = =(X_T482[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T483    X_T482  
//        a       448         1         1  
//       n1        28     13440     12544  
//       n2        28       480       448  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 448, 28, 28 }
// Out stride: { 1, 13440, 480 }
// Input 1 offset: 0
// Input 1 stride: { 1, 12544, 448 }
// Tile size: { 448, 2, 2 }
// Contraction output var shape: fp32(1, 28, 28, 480):(376320, 13440, 480, 1):1470 KiB
// Computed true ops: 702464
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 7176
// Computed out regs: 7168
// Computed mem read: 7168
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 14, 1
__kernel void kernel_c68_sdk_156(__global float* restrict  X_T483, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[7] = {0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[1794];
  int n2_gid = (get_group_id(0) * 2);
  int n1_gid = (get_group_id(1) * 2);
  {
    {
      int gbase = ((n2_gid * 448) + (n1_gid * 12544));
      int a_n2_tid = (tid % 256);
      for (int a_n2_lid = 0; a_n2_lid < 4; a_n2_lid += 1)
      {
        int a_n2_cond = ((a_n2_lid < 3) || (a_n2_tid < 128));
        if (a_n2_cond)
        {
          int a_n2 = ((256 * a_n2_lid) + a_n2_tid);
          for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
          {
            int lidx = (a_n2 + (897 * n1_lid));
            int gidx = ((gbase + a_n2) + (12544 * n1_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)351231)];
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 64);
    int n2_tid = ((tid / 64) % 2);
    int n1_tid = ((tid / 128) % 2);
    for (int a_lid = 0; a_lid < 7; a_lid += 1)
    {
      int a = ((64 * a_lid) + a_tid);
      float val1 = in1_shared[((a + (448 * n2_tid)) + (897 * n1_tid))];
      agg[a_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 64);
  int n2_tid = ((tid / 64) % 2);
  int n1_tid = ((tid / 128) % 2);
  for (int a_lid = 0; a_lid < 7; a_lid += 1)
  {
    int a = ((64 * a_lid) + a_tid);
    float LX_T483 = agg[a_lid];
    int gout_idx = ((a + (13440 * (n1_gid + n1_tid))) + (480 * (n2_gid + n2_tid)));
    X_T483[gout_idx] = LX_T483;
  }
}
