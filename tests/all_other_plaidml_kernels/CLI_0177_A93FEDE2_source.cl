#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 8, 8, 224 }
// Out stride: { 14336, 1792, 224, 1 }
// Elementwise input X_T2028 shape: fp32(1, 8, 8, 224):(14336, 1792, 224, 1):56 KiB
// Elementwise input X_T2032 shape: fp32(224):(1):896 bytes
// Elementwise input X_I_727 shape: fp32(224):(1):896 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T2033 = div(X_T2028, X_T2032)
// Elementwise op: [[pid(Add, Switch)]] X_T2034 = add(X_T2033, X_I_727)
// Elementwise op: X_T2035 = cmp_lt(X_T2034, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2036 = cond(X_T2035, X_T2, X_T2034)
// Tile size: { 1, 4, 1, 224 }
// Contraction output var shape: fp32(1, 8, 8, 224):(14336, 1792, 224, 1):56 KiB
// Computed true ops: 57344
// Computed work groups: 16
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 512, 8, 1
__kernel void kernel_c51_sdk_664(__global float* restrict  X_T2036, __global const float* restrict  X_T2028, __global const float* restrict  X_T2032, __global const float* restrict  X_I_727)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_tid < 32));
    if (i4_cond)
    {
      int i4 = ((64 * i4_lid) + i4_tid);
      int gout_idx = (((1792 * (i2_gid + i2_tid)) + (224 * i3_gid)) + i4);
      float LX_T2028 = X_T2028[gout_idx];
      float LX_T2032 = X_T2032[i4];
      float LX_I_727 = X_I_727[i4];
      float LX_T2033 = (LX_T2028 / LX_T2032);
      float LX_T2034 = (LX_T2033 + LX_I_727);
      int LX_T2035 = (LX_T2034 < 0.0f);
      float LX_T2036 = select((float)LX_T2034, (float)0.0f, (int)LX_T2035);
      X_T2036[gout_idx] = LX_T2036;
    }
  }
}
