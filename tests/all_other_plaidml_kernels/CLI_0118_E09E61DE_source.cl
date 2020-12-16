#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 56 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 96 }
// Out stride: { 301056, 5376, 96, 1 }
// Elementwise input X_T89 shape: fp32(1, 56, 56, 96):(301056, 5376, 96, 1):1176 KiB
// Elementwise input X_T93 shape: fp32(96):(1):384 bytes
// Elementwise input X_I_38 shape: fp32(96):(1):384 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T94 = div(X_T89, X_T93)
// Elementwise op: [[pid(Add, Switch)]] X_T95 = add(X_T94, X_I_38)
// Elementwise op: X_T96 = cmp_lt(X_T95, X_T2)
// Elementwise op: [[pid(Relu)]] X_T97 = cond(X_T96, X_T2, X_T95)
// Tile size: { 1, 8, 1, 96 }
// Contraction output var shape: fp32(1, 56, 56, 96):(301056, 5376, 96, 1):1176 KiB
// Computed true ops: 1204224
// Computed work groups: 392
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 3072
// Computed mem read: 288
// Computed mem write: 3072
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 56, 1
__kernel void kernel_c68_sdk_17(__global float* restrict  X_T97, __global const float* restrict  X_T89, __global const float* restrict  X_T93, __global const float* restrict  X_I_38)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 8);
  int i4_tid = (tid % 32);
  int i2_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 3; i4_lid += 1)
  {
    int i4 = ((32 * i4_lid) + i4_tid);
    int gout_idx = (((5376 * (i2_gid + i2_tid)) + (96 * i3_gid)) + i4);
    float LX_T89 = X_T89[gout_idx];
    float LX_T93 = X_T93[i4];
    float LX_I_38 = X_I_38[i4];
    float LX_T94 = (LX_T89 / LX_T93);
    float LX_T95 = (LX_T94 + LX_I_38);
    int LX_T96 = (LX_T95 < 0.0f);
    float LX_T97 = select((float)LX_T95, (float)0.0f, (int)LX_T96);
    X_T97[gout_idx] = LX_T97;
  }
}
