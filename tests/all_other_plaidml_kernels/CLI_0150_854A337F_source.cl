#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 384 }
// Out stride: { 1 }
// Elementwise input X_I_134 shape: fp32(384):(1):1.5 KiB
// Elementwise op: [[pid(Add)]] X_T340 = add(X_T33, X_I_134)
// Elementwise op: X_T341 = cmp_lt(X_T340, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T342 = cond(X_T341, X_T4, X_T340)
// Elementwise op: [[pid(Sqrt)]] X_T343 = sqrt(X_T342)
// Tile size: { 256 }
// Contraction output var shape: fp32(384):(1):1.5 KiB
// Computed true ops: 1536
// Computed work groups: 2
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 512, 1, 1
__kernel void kernel_c56_sdk_111(__global float* restrict  X_T343, __global const float* restrict  X_I_134)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 256) || (i1_tid < 128));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_134 = X_I_134[gout_idx];
    float LX_T340 = (0.0010000000474974513f + LX_I_134);
    int LX_T341 = (LX_T340 < (float)0);
    float LX_T342 = select((float)LX_T340, (float)0, (int)LX_T341);
    float LX_T343 = native_sqrt(LX_T342);
    X_T343[gout_idx] = LX_T343;
  }
}
