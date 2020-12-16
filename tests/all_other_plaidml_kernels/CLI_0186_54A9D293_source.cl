#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 320 }
// Out stride: { 1 }
// Elementwise input X_I_296 shape: fp32(320):(1):1.25 KiB
// Elementwise op: [[pid(Add)]] X_T843 = add(X_T33, X_I_296)
// Elementwise op: X_T844 = cmp_lt(X_T843, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T845 = cond(X_T844, X_T4, X_T843)
// Elementwise op: [[pid(Sqrt)]] X_T846 = sqrt(X_T845)
// Tile size: { 256 }
// Contraction output var shape: fp32(320):(1):1.25 KiB
// Computed true ops: 1280
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
__kernel void kernel_c56_sdk_286(__global float* restrict  X_T846, __global const float* restrict  X_I_296)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 256) || (i1_tid < 64));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_296 = X_I_296[gout_idx];
    float LX_T843 = (0.0010000000474974513f + LX_I_296);
    int LX_T844 = (LX_T843 < (float)0);
    float LX_T845 = select((float)LX_T843, (float)0, (int)LX_T844);
    float LX_T846 = native_sqrt(LX_T845);
    X_T846[gout_idx] = LX_T846;
  }
}
