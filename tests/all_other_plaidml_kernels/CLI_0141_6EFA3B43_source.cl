#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 8960 4 1
// lid: 256 1 1
// Original:
// X_T242[n, o0, o1, c : _T306, _T307, _T308, _T309] = =(X_T160[])
// With Index Variables Made Integral:
// X_T242[n, o0, o1, c : _T306, _T307, _T308, _T309] = =(X_T160[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 35, 0 <= o1 < 35, 0 <= c < 256, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 35, 0 <= o1 < 35, 0 <= c < 256 }
// Defracted:
// X_T242[n, o0, o1, c : _T306, _T307, _T308, _T309] = =(X_T160[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range    X_T242    X_T160  
//        c       256         1         0  
//       o0        35      8960         0  
//       o1        35       256         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 256, 35, 35 }
// Out stride: { 1, 8960, 256 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 64, 1, 35 }
// Contraction output var shape: fp32(1, 35, 35, 256):(313600, 8960, 256, 1):1225 KiB
// Computed true ops: 627200
// Computed work groups: 140
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 10240
// Computed mem read: 128
// Computed mem write: 8960
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 8960, 4, 1
__kernel void kernel_c56_sdk_72(__global float* restrict  X_T242)
{
  int tid = get_local_id(0);
  float agg[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  int c_gid = (get_group_id(1) * 64);
  int o0_gid = get_group_id(0);
  {
    int c_tid = (tid % 32);
    int o1_tid = ((tid / 32) % 8);
    for (int o1_lid = 0; o1_lid < 5; o1_lid += 1)
    {
      int o1_cond = ((o1_lid < 4) || (o1_tid < 3));
      int o1 = select((int)0, (int)((8 * o1_lid) + o1_tid), (int)o1_cond);
      for (int c_lid = 0; c_lid < 2; c_lid += 1)
      {
        int c = ((32 * c_lid) + c_tid);
        float val1 = 1.0f;
        int agg_idx = (c_lid + (o1_lid * 2));
        agg[agg_idx] = select((float)agg[agg_idx], (float)val1, (int)o1_cond);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o1_tid = ((tid / 32) % 8);
  for (int o1_lid = 0; o1_lid < 5; o1_lid += 1)
  {
    int o1_cond = ((o1_lid < 4) || (o1_tid < 3));
    if (o1_cond)
    {
      int o1 = ((8 * o1_lid) + o1_tid);
      for (int c_lid = 0; c_lid < 2; c_lid += 1)
      {
        int c = ((32 * c_lid) + c_tid);
        float LX_T242 = agg[(c_lid + (o1_lid * 2))];
        int gout_idx = (((c_gid + c) + (8960 * o0_gid)) + (256 * o1));
        X_T242[gout_idx] = LX_T242;
      }
    }
  }
}
