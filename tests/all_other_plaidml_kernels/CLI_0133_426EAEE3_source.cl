#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Original:
// X_T421[n, o0, o1, c : _T627, _T628, _T629, _T630] = =(X_T97[])
// With Index Variables Made Integral:
// X_T421[n, o0, o1, c : _T627, _T628, _T629, _T630] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= c < 22, 0 <= o0 < 28, 0 <= o1 < 28, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= c < 22, 0 <= o0 < 28, 0 <= o1 < 28 }
// Defracted:
// X_T421[n, o0, o1, c : _T627, _T628, _T629, _T630] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range    X_T421     X_T97  
//        c        22         1         0  
//       o0        28       616         0  
//       o1        28        22         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 22, 28, 28 }
// Out stride: { 1, 616, 22 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 22, 28, 1 }
// Contraction output var shape: fp32(1, 28, 28, 22):(17248, 616, 22, 1):67.375 KiB
// Computed true ops: 34496
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 4096
// Computed mem read: 128
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c42_sdk_144(__global float* restrict  X_T421)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  int o1_gid = get_group_id(0);
  {
    int c_tid = (tid % 32);
    int o0_tid = ((tid / 32) % 8);
    int c_cond = (c_tid < 22);
    int c = select((int)0, (int)c_tid, (int)c_cond);
    for (int o0_lid = 0; o0_lid < 4; o0_lid += 1)
    {
      int o0_cond = ((o0_lid < 3) || (o0_tid < 4));
      int o0 = select((int)0, (int)((8 * o0_lid) + o0_tid), (int)(c_cond && o0_cond));
      float val1 = 1.0f;
      agg[o0_lid] = select((float)agg[o0_lid], (float)val1, (int)(c_cond && o0_cond));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o0_tid = ((tid / 32) % 8);
  int c_cond = (c_tid < 22);
  if (c_cond)
  {
    for (int o0_lid = 0; o0_lid < 4; o0_lid += 1)
    {
      int o0_cond = ((o0_lid < 3) || (o0_tid < 4));
      if (o0_cond)
      {
        int o0 = ((8 * o0_lid) + o0_tid);
        float LX_T421 = agg[o0_lid];
        int gout_idx = ((c_tid + (616 * o0)) + (22 * o1_gid));
        X_T421[gout_idx] = LX_T421;
      }
    }
  }
}
