#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 4 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 128 }
// Out stride: { 6272, 896, 128, 1 }
// Elementwise input X_T1406 shape: fp32(1, 7, 7, 128):(6272, 896, 128, 1):24.5 KiB
// Elementwise input X_T1410 shape: fp32(128):(1):512 bytes
// Elementwise input X_I_527 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T1411 = div(X_T1406, X_T1410)
// Elementwise op: [[pid(Add, Switch)]] X_T1412 = add(X_T1411, X_I_527)
// Elementwise op: X_T1413 = cmp_lt(X_T1412, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1414 = cond(X_T1413, X_T2, X_T1412)
// Tile size: { 1, 1, 2, 128 }
// Contraction output var shape: fp32(1, 7, 7, 128):(6272, 896, 128, 1):24.5 KiB
// Computed true ops: 25088
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 96
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 4, 1
__kernel void kernel_c108_sdk_479(__global float* restrict  X_T1414, __global const float* restrict  X_T1406, __global const float* restrict  X_T1410, __global const float* restrict  X_I_527)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(1) * 2);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 128);
  int i3_tid = ((tid / 128) % 2);
  int i3_cond = ((i3_gid != 6) || (i3_tid < 1));
  if (i3_cond)
  {
    int gout_idx = (((896 * i2_gid) + (128 * (i3_gid + i3_tid))) + i4_tid);
    float LX_T1406 = X_T1406[gout_idx];
    float LX_T1410 = X_T1410[i4_tid];
    float LX_I_527 = X_I_527[i4_tid];
    float LX_T1411 = (LX_T1406 / LX_T1410);
    float LX_T1412 = (LX_T1411 + LX_I_527);
    int LX_T1413 = (LX_T1412 < 0.0f);
    float LX_T1414 = select((float)LX_T1412, (float)0.0f, (int)LX_T1413);
    X_T1414[gout_idx] = LX_T1414;
  }
}
