#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 14 1
// lid: 256 1 1
// Original:
// X_T653[n0, n1, n2, a : _T908, _T909, _T910, _T911] = =(X_T652[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T653[n0, n1, n2, a : _T908, _T909, _T910, _T911] = =(X_T652[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 384, 0 <= a < 416, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 384 }
// Defracted:
// X_T653[n0, n1, n2, a : _T908, _T909, _T910, _T911] = =(X_T652[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T653    X_T652  
//        a       384         1         1  
//       n1        14      5824      5376  
//       n2        14       416       384  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 384, 14, 14 }
// Out stride: { 1, 5824, 416 }
// Input 1 offset: 0
// Input 1 stride: { 1, 5376, 384 }
// Tile size: { 384, 2, 1 }
// Contraction output var shape: fp32(1, 14, 14, 416):(81536, 5824, 416, 1):318.5 KiB
// Computed true ops: 150528
// Computed work groups: 98
// Computed inner loops: 1
// Computed shared mem: 3080
// Computed out regs: 3072
// Computed mem read: 3072
// Computed mem write: 3072
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 14, 1
__kernel void kernel_c68_sdk_216(__global float* restrict  X_T653, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[3] = {0, 0, 0, };
  __local float in1_shared[770];
  int n2_gid = get_group_id(1);
  int n1_gid = (get_group_id(0) * 2);
  {
    {
      int gbase = ((n2_gid * 384) + (n1_gid * 5376));
      int a_n2_tid = (tid % 256);
      for (int a_n2_lid = 0; a_n2_lid < 2; a_n2_lid += 1)
      {
        int a_n2_cond = ((a_n2_lid < 1) || (a_n2_tid < 128));
        if (a_n2_cond)
        {
          int a_n2 = ((256 * a_n2_lid) + a_n2_tid);
          for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
          {
            int lidx = (a_n2 + (385 * n1_lid));
            int gidx = ((gbase + a_n2) + (5376 * n1_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)75263)];
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 128);
    int n1_tid = ((tid / 128) % 2);
    for (int a_lid = 0; a_lid < 3; a_lid += 1)
    {
      int a = ((128 * a_lid) + a_tid);
      float val1 = in1_shared[(a + (385 * n1_tid))];
      agg[a_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 128);
  int n1_tid = ((tid / 128) % 2);
  for (int a_lid = 0; a_lid < 3; a_lid += 1)
  {
    int a = ((128 * a_lid) + a_tid);
    float LX_T653 = agg[a_lid];
    int gout_idx = ((a + (5824 * (n1_gid + n1_tid))) + (416 * n2_gid));
    X_T653[gout_idx] = LX_T653;
  }
}
