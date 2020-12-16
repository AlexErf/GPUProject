#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 56 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 128 }
// Out stride: { 401408, 7168, 128, 1 }
// Elementwise input X_T95 shape: fp32(1, 56, 56, 128):(401408, 7168, 128, 1):1568 KiB
// Elementwise input X_T99 shape: fp32(128):(1):512 bytes
// Elementwise input X_I_24 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T100 = div(X_T95, X_T99)
// Elementwise op: [[pid(Add, Switch)]] X_T101 = add(X_T100, X_I_24)
// Elementwise op: X_T102 = cmp_lt(X_T101, X_T2)
// Elementwise op: [[pid(Relu)]] X_T103 = cond(X_T102, X_T2, X_T101)
// Tile size: { 1, 4, 1, 128 }
// Contraction output var shape: fp32(1, 56, 56, 128):(401408, 7168, 128, 1):1568 KiB
// Computed true ops: 1605632
// Computed work groups: 784
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 192
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 56, 1
__kernel void kernel_c108_sdk_11(__global float* restrict  X_T103, __global const float* restrict  X_T95, __global const float* restrict  X_T99, __global const float* restrict  X_I_24)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
  {
    int i4 = ((64 * i4_lid) + i4_tid);
    int gout_idx = (((7168 * (i2_gid + i2_tid)) + (128 * i3_gid)) + i4);
    float LX_T95 = X_T95[gout_idx];
    float LX_T99 = X_T99[i4];
    float LX_I_24 = X_I_24[i4];
    float LX_T100 = (LX_T95 / LX_T99);
    float LX_T101 = (LX_T100 + LX_I_24);
    int LX_T102 = (LX_T101 < 0.0f);
    float LX_T103 = select((float)LX_T101, (float)0.0f, (int)LX_T102);
    X_T103[gout_idx] = LX_T103;
  }
}
