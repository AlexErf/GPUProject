#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 21 1
// lid: 256 1 1
// Original:
// X_T422[n, o0, o1, c : _T627, _T628, _T629, _T630] = =(X_T100[])
// With Index Variables Made Integral:
// X_T422[n, o0, o1, c : _T627, _T628, _T629, _T630] = =(X_T100[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 42, 0 <= o1 < 42, 0 <= c < 84, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 42, 0 <= o1 < 42, 0 <= c < 84 }
// Defracted:
// X_T422[n, o0, o1, c : _T627, _T628, _T629, _T630] = =(X_T100[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range    X_T422    X_T100  
//        c        84         1         0  
//       o0        42      3528         0  
//       o1        42        84         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 84, 42, 42 }
// Out stride: { 1, 3528, 84 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 84, 2, 2 }
// Contraction output var shape: fp32(1, 42, 42, 84):(148176, 3528, 84, 1):578.812 KiB
// Computed true ops: 296352
// Computed work groups: 441
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 2048
// Computed mem read: 128
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 21, 1
__kernel void kernel_c42_sdk_144(__global float* restrict  X_T422)
{
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  int o1_gid = (get_group_id(0) * 2);
  int o0_gid = (get_group_id(1) * 2);
  {
    int c_tid = (tid % 64);
    int o1_tid = ((tid / 64) % 2);
    int o0_tid = ((tid / 128) % 2);
    for (int c_lid = 0; c_lid < 2; c_lid += 1)
    {
      int c_cond = ((c_lid < 1) || (c_tid < 20));
      int c = select((int)0, (int)((64 * c_lid) + c_tid), (int)c_cond);
      float val1 = 1.0f;
      agg[c_lid] = select((float)agg[c_lid], (float)val1, (int)c_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 64);
  int o1_tid = ((tid / 64) % 2);
  int o0_tid = ((tid / 128) % 2);
  for (int c_lid = 0; c_lid < 2; c_lid += 1)
  {
    int c_cond = ((c_lid < 1) || (c_tid < 20));
    if (c_cond)
    {
      int c = ((64 * c_lid) + c_tid);
      float LX_T422 = agg[c_lid];
      int gout_idx = ((c + (3528 * (o0_gid + o0_tid))) + (84 * (o1_gid + o1_tid)));
      X_T422[gout_idx] = LX_T422;
    }
  }
}
