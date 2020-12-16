#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 9472 37 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 37, 37, 256 }
// Out stride: { 350464, 9472, 256, 1 }
// Elementwise input X_T197 shape: fp32(1, 37, 37, 256):(350464, 9472, 256, 1):1369 KiB
// Elementwise input X_T201 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_192 shape: fp32(256):(1):1 KiB
// Elementwise input X_T190 shape: fp32(1, 37, 37, 256):(350464, 9472, 256, 1):1369 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T202 = div(X_T197, X_T201)
// Elementwise op: [[pid(Add, Switch)]] X_T203 = add(X_T202, X_I_192)
// Elementwise op: [[pid(Add)]] X_T204 = add(X_T190, X_T203)
// Elementwise op: X_T205 = cmp_lt(X_T204, X_T2)
// Elementwise op: [[pid(Relu)]] X_T206 = cond(X_T205, X_T2, X_T204)
// Tile size: { 1, 1, 1, 256 }
// Contraction output var shape: fp32(1, 37, 37, 256):(350464, 9472, 256, 1):1369 KiB
// Computed true ops: 1752320
// Computed work groups: 1369
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 128
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 9472, 37, 1
__kernel void kernel_c28_sdk_63(__global float* restrict  X_T204, __global float* restrict  X_T206, __global const float* restrict  X_T197, __global const float* restrict  X_T201, __global const float* restrict  X_I_192, __global const float* restrict  X_T190)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  int gout_idx = (((9472 * i2_gid) + (256 * i3_gid)) + i4_tid);
  float LX_T197 = X_T197[gout_idx];
  float LX_T201 = X_T201[i4_tid];
  float LX_I_192 = X_I_192[i4_tid];
  float LX_T190 = X_T190[gout_idx];
  float LX_T202 = (LX_T197 / LX_T201);
  float LX_T203 = (LX_T202 + LX_I_192);
  float LX_T204 = (LX_T190 + LX_T203);
  int LX_T205 = (LX_T204 < 0.0f);
  float LX_T206 = select((float)LX_T204, (float)0.0f, (int)LX_T205);
  X_T204[gout_idx] = LX_T204;
  X_T206[gout_idx] = LX_T206;
}
