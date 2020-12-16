#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Original:
// X_T200[n, x0, x1, c : _T262, _T263, _T264, _T265] = +(X_T199[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T200[n, x0, x1, c : _T262, _T263, _T264, _T265] = +(X_T199[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= c < 11, 0 <= c < 11, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= -1 + k0 + x0 < 56, 0 <= -1 + k1 + x1 < 56, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= c < 11, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= -1 + k0 + x0 < 56, 0 <= -1 + k1 + x1 < 56 }
// Defracted:
// X_T200[n, x0, x1, c : _T262, _T263, _T264, _T265] = +(X_T199[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T200    X_T199  
//        c        11         1         1  
//       k0         3         0       616  
//       k1         3         0        11  
//       x0        56       616       616  
//       x1        56        11        11  
//      off                   0      -627  
//      vec                   1         1  
// Constraint: (0,-1,0,-1,0) <= -1
// Constraint: (0,1,0,1,0) <= 56
// Constraint: (0,0,-1,0,-1) <= -1
// Constraint: (0,0,1,0,1) <= 56
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 11, 3, 3, 56, 56 }
// Out stride: { 1, 0, 0, 616, 11 }
// Input 1 offset: -627
// Input 1 stride: { 1, 616, 11, 616, 11 }
// Elementwise input X_T198 shape: fp32(1, 56, 56, 11):(34496, 616, 11, 1):134.75 KiB
// Elementwise input X_T94 shape: fp32(1, 56, 56, 11):(34496, 616, 11, 1):134.75 KiB
// Elementwise op: [[pid(reduction_A_block_stem_1)]] X_T201 = div(X_T198, X_T200)
// Elementwise op: [[pid(Add)]] X_T202 = add(X_T94, X_T201)
// Tile size: { 11, 3, 3, 8, 8 }
// Contraction output var shape: fp32(1, 56, 56, 11):(34496, 616, 11, 1):134.75 KiB
// Computed true ops: 1241856
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 4440
// Computed out regs: 4096
// Computed mem read: 4864
// Computed mem write: 8192
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c42_sdk_59(__global float* restrict  X_T202, __global const float* restrict  in1, __global const float* restrict  X_T198, __global const float* restrict  X_T94)
{
  in1 = (in1 + -627);
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[1110];
  int x1_gid = (get_group_id(0) * 8);
  int x0_gid = (get_group_id(1) * 8);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((((k1_gid * 11) + (x1_gid * 11)) + (k0_gid * 616)) + (x0_gid * 616));
        int c_k1_x1_tid = (tid % 128);
        int k0_x0_tid = ((tid / 128) % 2);
        int c_k1_x1_cond = (c_k1_x1_tid < 110);
        if (c_k1_x1_cond)
        {
          for (int k0_x0_lid = 0; k0_x0_lid < 5; k0_x0_lid += 1)
          {
            int k0_x0 = ((2 * k0_x0_lid) + k0_x0_tid);
            int lidx = (c_k1_x1_tid + (111 * k0_x0));
            int gidx = ((gbase + c_k1_x1_tid) + (616 * k0_x0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)627, (int)35122)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -1) && ((((k0_gid + 3) - 1) + ((x0_gid + 8) - 1)) <= 56)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -1)) && ((((k1_gid + 3) - 1) + ((x1_gid + 8) - 1)) <= 56)))
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int c_tid = (tid % 16);
            int x1_tid = ((tid / 16) % 4);
            int x0_tid = ((tid / 64) % 4);
            int c_cond = (c_tid < 11);
            int c = select((int)0, (int)c_tid, (int)c_cond);
            for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
            {
              int x1 = ((4 * x1_lid) + x1_tid);
              for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
              {
                int x0 = ((4 * x0_lid) + x0_tid);
                float val1 = in1_shared[((((c + (11 * k1_lid)) + (11 * x1)) + (111 * k0_lid)) + (111 * x0))];
                int agg_idx = (x1_lid + (x0_lid * 2));
                float agg_rhs = (agg[agg_idx] + val1);
                agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)c_cond);
              }
            }
          }
        }
      }
      else
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int c_tid = (tid % 16);
            int x1_tid = ((tid / 16) % 4);
            int x0_tid = ((tid / 64) % 4);
            int c_cond = (c_tid < 11);
            int c = select((int)0, (int)c_tid, (int)c_cond);
            for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
            {
              int x1 = ((4 * x1_lid) + x1_tid);
              for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
              {
                int x0 = ((4 * x0_lid) + x0_tid);
                float val1 = in1_shared[((((c + (11 * k1_lid)) + (11 * x1)) + (111 * k0_lid)) + (111 * x0))];
                int agg_idx = (x1_lid + (x0_lid * 2));
                float agg_rhs = (agg[agg_idx] + val1);
                agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)(c_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * (x0_gid + x0))) <= -1) && (((k0_gid + k0_lid) + (x0_gid + x0)) <= 56)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1))) <= -1)) && (((k1_gid + k1_lid) + (x1_gid + x1)) <= 56))));
              }
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 16);
  int x1_tid = ((tid / 16) % 4);
  int x0_tid = ((tid / 64) % 4);
  int c_cond = (c_tid < 11);
  if (c_cond)
  {
    for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
      {
        int x0 = ((4 * x0_lid) + x0_tid);
        float LX_T200 = agg[(x1_lid + (x0_lid * 2))];
        int gout_idx = ((c_tid + (616 * (x0_gid + x0))) + (11 * (x1_gid + x1)));
        if (((gout_idx >= 0) && (gout_idx < 34496)))
        {
          float LX_T198 = X_T198[gout_idx];
          float LX_T94 = X_T94[gout_idx];
          float LX_T201 = (LX_T198 / LX_T200);
          float LX_T202 = (LX_T94 + LX_T201);
          X_T202[gout_idx] = LX_T202;
        }
      }
    }
  }
}
