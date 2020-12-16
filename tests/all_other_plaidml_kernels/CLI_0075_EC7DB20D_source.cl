#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 11 }
// Out stride: { 34496, 616, 11, 1 }
// Elementwise input X_T74 shape: fp32(1, 56, 56, 11):(34496, 616, 11, 1):134.75 KiB
// Elementwise input X_T78 shape: fp32(11):(1):44 bytes
// Elementwise input X_I_54 shape: fp32(11):(1):44 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T79 = div(X_T74, X_T78)
// Elementwise op: [[pid(Add, Switch)]] X_T80 = add(X_T79, X_I_54)
// Elementwise op: X_T81 = cmp_lt(X_T80, X_T1)
// Elementwise op: [[pid(Relu)]] X_T82 = cond(X_T81, X_T1, X_T80)
// Tile size: { 1, 56, 2, 11 }
// Contraction output var shape: fp32(1, 56, 56, 11):(34496, 616, 11, 1):134.75 KiB
// Computed true ops: 137984
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 1344
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c42_sdk_14(__global float* restrict  X_T82, __global const float* restrict  X_T74, __global const float* restrict  X_T78, __global const float* restrict  X_I_54)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 16);
  int i3_tid = ((tid / 16) % 2);
  int i2_tid = ((tid / 32) % 8);
  int i4_cond = (i4_tid < 11);
  if (i4_cond)
  {
    for (int i2_lid = 0; i2_lid < 7; i2_lid += 1)
    {
      int i2 = ((8 * i2_lid) + i2_tid);
      int gout_idx = (((616 * i2) + (11 * (i3_gid + i3_tid))) + i4_tid);
      float LX_T74 = X_T74[gout_idx];
      float LX_T78 = X_T78[i4_tid];
      float LX_I_54 = X_I_54[i4_tid];
      float LX_T79 = (LX_T74 / LX_T78);
      float LX_T80 = (LX_T79 + LX_I_54);
      int LX_T81 = (LX_T80 < 0.0f);
      float LX_T82 = select((float)LX_T80, (float)0.0f, (int)LX_T81);
      X_T82[gout_idx] = LX_T82;
    }
  }
}
