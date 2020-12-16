#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 14 1
// lid: 256 1 1
// Original:
// X_T215[n0, n1, n2, a : _T216, _T217, _T218, _T219] = =(X_T214[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T215[n0, n1, n2, a : _T216, _T217, _T218, _T219] = =(X_T214[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= a < 224, 0 <= a < 256, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= a < 224 }
// Defracted:
// X_T215[n0, n1, n2, a : _T216, _T217, _T218, _T219] = =(X_T214[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T215    X_T214  
//        a       224         1         1  
//       n1        56     14336     12544  
//       n2        56       256       224  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 224, 56, 56 }
// Out stride: { 1, 14336, 256 }
// Input 1 offset: 0
// Input 1 stride: { 1, 12544, 224 }
// Tile size: { 224, 4, 4 }
// Contraction output var shape: fp32(1, 56, 56, 256):(802816, 14336, 256, 1):3136 KiB
// Computed true ops: 1404928
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 14352
// Computed out regs: 14336
// Computed mem read: 14336
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 14, 1
__kernel void kernel_c124_sdk_51(__global float* restrict  X_T215, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3588];
  int n2_gid = (get_group_id(0) * 4);
  int n1_gid = (get_group_id(1) * 4);
  {
    {
      int gbase = ((n2_gid * 224) + (n1_gid * 12544));
      int a_n2_tid = (tid % 256);
      for (int a_n2_lid = 0; a_n2_lid < 4; a_n2_lid += 1)
      {
        int a_n2_cond = ((a_n2_lid < 3) || (a_n2_tid < 128));
        if (a_n2_cond)
        {
          int a_n2 = ((256 * a_n2_lid) + a_n2_tid);
          for (int n1_lid = 0; n1_lid < 4; n1_lid += 1)
          {
            int lidx = (a_n2 + (897 * n1_lid));
            int gidx = ((gbase + a_n2) + (12544 * n1_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)702463)];
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 4);
    int n1_tid = ((tid / 128) % 2);
    for (int a_lid = 0; a_lid < 7; a_lid += 1)
    {
      int a = ((32 * a_lid) + a_tid);
      for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
      {
        int n1 = ((2 * n1_lid) + n1_tid);
        float val1 = in1_shared[((a + (224 * n2_tid)) + (897 * n1))];
        int agg_idx = (a_lid + (n1_lid * 7));
        agg[agg_idx] = val1;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 4);
  int n1_tid = ((tid / 128) % 2);
  for (int a_lid = 0; a_lid < 7; a_lid += 1)
  {
    int a = ((32 * a_lid) + a_tid);
    for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
    {
      int n1 = ((2 * n1_lid) + n1_tid);
      float LX_T215 = agg[(a_lid + (n1_lid * 7))];
      int gout_idx = ((a + (14336 * (n1_gid + n1))) + (256 * (n2_gid + n2_tid)));
      X_T215[gout_idx] = LX_T215;
    }
  }
}
