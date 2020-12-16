#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Original:
// X_T526[n0, n1, n2, 448 + a : _T687, _T688, _T689, _T690] = =(X_T525[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T526[n0, n1, n2, 448 + a : _T687, _T688, _T689, _T690] = =(X_T525[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= a < 32, 0 <= 448 + a < 480, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= a < 32 }
// Defracted:
// X_T526[n0, n1, n2, 448 + a : _T687, _T688, _T689, _T690] = =(X_T525[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T526    X_T525  
//        a        32         1         1  
//       n1        28     13440       896  
//       n2        28       480        32  
//      off                 448         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 32, 28, 28 }
// Out stride: { 1, 13440, 480 }
// Input 1 offset: 0
// Input 1 stride: { 1, 896, 32 }
// Tile size: { 32, 28, 1 }
// Contraction output var shape: fp32(1, 28, 28, 480):(376320, 13440, 480, 1):1470 KiB
// Computed true ops: 50176
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 3696
// Computed out regs: 4096
// Computed mem read: 3584
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c108_sdk_163(__global float* restrict  X_T526, __global const float* restrict  in1)
{
  X_T526 = (X_T526 + 448);
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[924];
  int n2_gid = get_group_id(0);
  {
    {
      int gbase = (n2_gid * 32);
      int a_n2_tid = (tid % 32);
      int n1_tid = ((tid / 32) % 8);
      for (int n1_lid = 0; n1_lid < 4; n1_lid += 1)
      {
        int n1_cond = ((n1_lid < 3) || (n1_tid < 4));
        if (n1_cond)
        {
          int n1 = ((8 * n1_lid) + n1_tid);
          int lidx = (a_n2_tid + (33 * n1));
          int gidx = ((gbase + a_n2_tid) + (896 * n1));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)25087)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n1_tid = ((tid / 32) % 8);
    for (int n1_lid = 0; n1_lid < 4; n1_lid += 1)
    {
      int n1_cond = ((n1_lid < 3) || (n1_tid < 4));
      int n1 = select((int)0, (int)((8 * n1_lid) + n1_tid), (int)n1_cond);
      float val1 = in1_shared[(a_tid + (33 * n1))];
      agg[n1_lid] = select((float)agg[n1_lid], (float)val1, (int)n1_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n1_tid = ((tid / 32) % 8);
  for (int n1_lid = 0; n1_lid < 4; n1_lid += 1)
  {
    int n1_cond = ((n1_lid < 3) || (n1_tid < 4));
    if (n1_cond)
    {
      int n1 = ((8 * n1_lid) + n1_tid);
      float LX_T526 = agg[n1_lid];
      int gout_idx = ((a_tid + (13440 * n1)) + (480 * n2_gid));
      X_T526[gout_idx] = LX_T526;
    }
  }
}
