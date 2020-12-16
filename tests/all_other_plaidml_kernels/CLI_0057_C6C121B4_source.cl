#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 56 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 128 }
// Out stride: { 401408, 7168, 128, 1 }
// Elementwise input X_T132 shape: fp32(1, 56, 56, 128):(401408, 7168, 128, 1):1568 KiB
// Elementwise input X_T136 shape: fp32(128):(1):512 bytes
// Elementwise input X_I_91 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T137 = div(X_T132, X_T136)
// Elementwise op: [[pid(Add, Switch)]] X_T138 = add(X_T137, X_I_91)
// Elementwise op: X_T139 = cmp_lt(X_T138, X_T10)
// Elementwise op: [[pid(Relu)]] X_T140 = cond(X_T139, X_T10, X_T138)
// Elementwise op: X_T141 = cmp_lt(X_T140, X_T9)
// Elementwise op: [[pid(Relu)]] X_T142 = cond(X_T141, X_T140, X_T9)
// Tile size: { 1, 4, 1, 128 }
// Contraction output var shape: fp32(1, 56, 56, 128):(401408, 7168, 128, 1):1568 KiB
// Computed true ops: 2408448
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
__kernel void kernel_c25_sdk_32(__global float* restrict  X_T142, __global const float* restrict  X_T132, __global const float* restrict  X_T136, __global const float* restrict  X_I_91)
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
    float LX_T132 = X_T132[gout_idx];
    float LX_T136 = X_T136[i4];
    float LX_I_91 = X_I_91[i4];
    float LX_T137 = (LX_T132 / LX_T136);
    float LX_T138 = (LX_T137 + LX_I_91);
    int LX_T139 = (LX_T138 < 0.0f);
    float LX_T140 = select((float)LX_T138, (float)0.0f, (int)LX_T139);
    int LX_T141 = (LX_T140 < 6.0f);
    float LX_T142 = select((float)6.0f, (float)LX_T140, (int)LX_T141);
    X_T142[gout_idx] = LX_T142;
  }
}
