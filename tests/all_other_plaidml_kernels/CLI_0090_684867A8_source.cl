#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 9472 147 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 147, 147, 64 }
// Out stride: { 1382976, 9408, 64, 1 }
// Elementwise input X_T53 shape: fp32(1, 147, 147, 64):(1382976, 9408, 64, 1):5402.25 KiB
// Elementwise input X_T57 shape: fp32(64):(1):256 bytes
// Elementwise input X_I_23 shape: fp32(64):(1):256 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T58 = div(X_T53, X_T57)
// Elementwise op: [[pid(Add, Switch)]] X_T59 = add(X_T58, X_I_23)
// Elementwise op: X_T60 = cmp_lt(X_T59, X_T2)
// Elementwise op: [[pid(Relu)]] X_T61 = cond(X_T60, X_T2, X_T59)
// Tile size: { 1, 1, 4, 64 }
// Contraction output var shape: fp32(1, 147, 147, 64):(1382976, 9408, 64, 1):5402.25 KiB
// Computed true ops: 5531904
// Computed work groups: 5439
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 96
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 9472, 147, 1
__kernel void kernel_c51_sdk_8(__global float* restrict  X_T61, __global const float* restrict  X_T53, __global const float* restrict  X_T57, __global const float* restrict  X_I_23)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 4);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 4);
  int i3_cond = ((i3_gid != 144) || (i3_tid < 3));
  if (i3_cond)
  {
    int gout_idx = (((9408 * i2_gid) + (64 * (i3_gid + i3_tid))) + i4_tid);
    float LX_T53 = X_T53[gout_idx];
    float LX_T57 = X_T57[i4_tid];
    float LX_I_23 = X_I_23[i4_tid];
    float LX_T58 = (LX_T53 / LX_T57);
    float LX_T59 = (LX_T58 + LX_I_23);
    int LX_T60 = (LX_T59 < 0.0f);
    float LX_T61 = select((float)LX_T59, (float)0.0f, (int)LX_T60);
    X_T61[gout_idx] = LX_T61;
  }
}
