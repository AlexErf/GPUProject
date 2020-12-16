#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7936 165 1
// lid: 256 1 1
// Original:
// X_T113[n, d0, d1, c : _T117, _T118, _T119, _T120] = =(X_T48[n, -2 + d0, -2 + d1, c])
// With Index Variables Made Integral:
// X_T113[n, d0, d1, c : _T117, _T118, _T119, _T120] = =(X_T48[n, -2 + d0, -2 + d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= c < 96, 0 <= c < 96, 0 <= -2 + d0 < 165, 0 <= -2 + d1 < 165, 0 <= d0 < 169, 0 <= d1 < 169, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= c < 96, 0 <= -2 + d0 < 165, 0 <= -2 + d1 < 165 }
// Defracted:
// X_T113[n, d0, d1, c : _T117, _T118, _T119, _T120] = =(X_T48[n, -2 + d0, -2 + d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range    X_T113     X_T48  
//        c        96         1         1  
//       d0       165     16224     15840  
//       d1       165        96        96  
//      off               32640         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 15840, 165 }
// Out stride: { 1, 16224 }
// Input 1 offset: 0
// Input 1 stride: { 1, 15840 }
// Tile size: { 512, 1 }
// Contraction output var shape: fp32(1, 169, 169, 96):(2741856, 16224, 96, 1):10710.4 KiB
// Computed true ops: 5227200
// Computed work groups: 5115
// Computed inner loops: 1
// Computed shared mem: 2048
// Computed out regs: 2048
// Computed mem read: 2048
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7936, 165, 1
__kernel void kernel_c42_sdk_25(__global float* restrict  X_T113, __global const float* restrict  in1)
{
  X_T113 = (X_T113 + 32640);
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[512];
  int d1_c_gid = (get_group_id(0) * 512);
  int d0_gid = get_group_id(1);
  {
    {
      int gbase = (d1_c_gid + (d0_gid * 15840));
      int d1_c_tid = (tid % 256);
      for (int d1_c_lid = 0; d1_c_lid < 2; d1_c_lid += 1)
      {
        int d1_c = ((256 * d1_c_lid) + d1_c_tid);
        int gidx = (gbase + d1_c);
        in1_shared[d1_c] = in1[clamp((int)gidx, (int)0, (int)2613599)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int d1_c_tid = (tid % 256);
    for (int d1_c_lid = 0; d1_c_lid < 2; d1_c_lid += 1)
    {
      int d1_c = ((256 * d1_c_lid) + d1_c_tid);
      float val1 = in1_shared[d1_c];
      agg[d1_c_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int d1_c_tid = (tid % 256);
  for (int d1_c_lid = 0; d1_c_lid < 2; d1_c_lid += 1)
  {
    int d1_c_cond = ((d1_c_lid < 1) || ((d1_c_gid != 15360) || (d1_c_tid < 224)));
    if (d1_c_cond)
    {
      int d1_c = ((256 * d1_c_lid) + d1_c_tid);
      float LX_T113 = agg[d1_c_lid];
      int gout_idx = ((d1_c_gid + d1_c) + (16224 * d0_gid));
      X_T113[gout_idx] = LX_T113;
    }
  }
}
