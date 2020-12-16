#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 4 4
// lid: 256 1 1
// Original:
// X_T894[n, x0, x1, c : _T1255, _T1256, _T1257, _T1258] = >(X_T830[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T894[n, x0, x1, c : _T1255, _T1256, _T1257, _T1258] = >(X_T830[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 8, 0 <= x1 < 8, 0 <= k0 + 2*x0 < 17, 0 <= k1 + 2*x1 < 17, 0 <= c < 768, 0 <= c < 768, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 8, 0 <= x1 < 8, 0 <= k0 + 2*x0 < 17, 0 <= k1 + 2*x1 < 17, 0 <= c < 768 }
// Defracted:
// X_T894[n, x0, x1, c : _T1255, _T1256, _T1257, _T1258] = >(X_T830[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T894    X_T830  
//        c       768         1         1  
//       k0         3         0     13056  
//       k1         3         0       768  
//       x0         8      6144     26112  
//       x1         8       768      1536  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 768, 3, 3, 8, 8 }
// Out stride: { 1, 0, 0, 6144, 768 }
// Input 1 offset: 0
// Input 1 stride: { 1, 13056, 768, 26112, 1536 }
// Tile size: { 256, 3, 3, 2, 2 }
// Contraction output var shape: fp32(1, 8, 8, 768):(49152, 6144, 768, 1):192 KiB
// Computed true ops: 884736
// Computed work groups: 48
// Computed inner loops: 1
// Computed shared mem: 25600
// Computed out regs: 4096
// Computed mem read: 25600
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 4, 4
__kernel void kernel_c56_sdk_303(__global float* restrict  X_T894, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, };
  __local float in1_shared[6400];
  int c_gid = (get_group_id(0) * 256);
  int x1_gid = (get_group_id(1) * 2);
  int x0_gid = (get_group_id(2) * 2);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((((c_gid + (k1_gid * 768)) + (x1_gid * 1536)) + (k0_gid * 13056)) + (x0_gid * 26112));
        int c_tid = (tid % 256);
        for (int k1_x1_lid = 0; k1_x1_lid < 5; k1_x1_lid += 1)
        {
          for (int k0_x0_lid = 0; k0_x0_lid < 5; k0_x0_lid += 1)
          {
            int lidx = (((25 * c_tid) + k1_x1_lid) + (5 * k0_x0_lid));
            int gidx = (((gbase + c_tid) + (768 * k1_x1_lid)) + (13056 * k0_x0_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)221951)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
      {
        for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
        {
          int c_tid = (tid % 64);
          int x1_tid = ((tid / 64) % 2);
          int x0_tid = ((tid / 128) % 2);
          for (int c_lid = 0; c_lid < 4; c_lid += 1)
          {
            int c = ((64 * c_lid) + c_tid);
            float val1 = in1_shared[(((((25 * c) + k1_lid) + (2 * x1_tid)) + (5 * k0_lid)) + (10 * x0_tid))];
            float agg_rhs = select((float)agg[c_lid], (float)val1, (int)(val1 > agg[c_lid]));
            agg[c_lid] = agg_rhs;
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 64);
  int x1_tid = ((tid / 64) % 2);
  int x0_tid = ((tid / 128) % 2);
  for (int c_lid = 0; c_lid < 4; c_lid += 1)
  {
    int c = ((64 * c_lid) + c_tid);
    float LX_T894 = agg[c_lid];
    LX_T894 = select((float)LX_T894, (float)0, (int)(LX_T894 == (float)-FLT_MAX));
    int gout_idx = (((c_gid + c) + (6144 * (x0_gid + x0_tid))) + (768 * (x1_gid + x1_tid)));
    X_T894[gout_idx] = LX_T894;
  }
}
