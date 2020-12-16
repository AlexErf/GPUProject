#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 56 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 96 }
// Out stride: { 301056, 5376, 96, 1 }
// Elementwise input X_T111 shape: fp32(1, 56, 56, 96):(301056, 5376, 96, 1):1176 KiB
// Elementwise input X_T115 shape: fp32(96):(1):384 bytes
// Elementwise input X_I_70 shape: fp32(96):(1):384 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T116 = div(X_T111, X_T115)
// Elementwise op: [[pid(Add, Switch)]] X_T117 = add(X_T116, X_I_70)
// Elementwise op: X_T118 = cmp_lt(X_T117, X_T3)
// Elementwise op: [[pid(Relu)]] X_T119 = cond(X_T118, X_T3, X_T117)
// Elementwise op: X_T120 = cmp_lt(X_T119, X_T2)
// Elementwise op: [[pid(Relu)]] X_T121 = cond(X_T120, X_T119, X_T2)
// Tile size: { 1, 8, 1, 96 }
// Contraction output var shape: fp32(1, 56, 56, 96):(301056, 5376, 96, 1):1176 KiB
// Computed true ops: 1806336
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
__kernel void kernel_c43_sdk_25(__global float* restrict  X_T121, __global const float* restrict  X_T111, __global const float* restrict  X_T115, __global const float* restrict  X_I_70)
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
    float LX_T111 = X_T111[gout_idx];
    float LX_T115 = X_T115[i4];
    float LX_I_70 = X_I_70[i4];
    float LX_T116 = (LX_T111 / LX_T115);
    float LX_T117 = (LX_T116 + LX_I_70);
    int LX_T118 = (LX_T117 < 0.0f);
    float LX_T119 = select((float)LX_T117, (float)0.0f, (int)LX_T118);
    int LX_T120 = (LX_T119 < 6.0f);
    float LX_T121 = select((float)6.0f, (float)LX_T119, (int)LX_T120);
    X_T121[gout_idx] = LX_T121;
  }
}
