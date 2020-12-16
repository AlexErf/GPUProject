#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 9216 71 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 71, 71, 192 }
// Out stride: { 967872, 13632, 192, 1 }
// Elementwise input X_T75 shape: fp32(1, 71, 71, 192):(967872, 13632, 192, 1):3780.75 KiB
// Elementwise input X_T79 shape: fp32(192):(1):768 bytes
// Elementwise input X_I_17 shape: fp32(192):(1):768 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T80 = div(X_T75, X_T79)
// Elementwise op: [[pid(Add, Switch)]] X_T81 = add(X_T80, X_I_17)
// Elementwise op: X_T82 = cmp_lt(X_T81, X_T2)
// Elementwise op: [[pid(Relu)]] X_T83 = cond(X_T82, X_T2, X_T81)
// Tile size: { 1, 2, 1, 192 }
// Contraction output var shape: fp32(1, 71, 71, 192):(967872, 13632, 192, 1):3780.75 KiB
// Computed true ops: 3871488
// Computed work groups: 2556
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 144
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 9216, 71, 1
__kernel void kernel_c51_sdk_15(__global float* restrict  X_T83, __global const float* restrict  X_T75, __global const float* restrict  X_T79, __global const float* restrict  X_I_17)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 128);
  int i2_tid = ((tid / 128) % 2);
  int i2_cond = ((i2_gid != 70) || (i2_tid < 1));
  if (i2_cond)
  {
    for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
    {
      int i4_cond = ((i4_lid < 1) || (i4_tid < 64));
      if (i4_cond)
      {
        int i4 = ((128 * i4_lid) + i4_tid);
        int gout_idx = (((13632 * (i2_gid + i2_tid)) + (192 * i3_gid)) + i4);
        float LX_T75 = X_T75[gout_idx];
        float LX_T79 = X_T79[i4];
        float LX_I_17 = X_I_17[i4];
        float LX_T80 = (LX_T75 / LX_T79);
        float LX_T81 = (LX_T80 + LX_I_17);
        int LX_T82 = (LX_T81 < 0.0f);
        float LX_T83 = select((float)LX_T81, (float)0.0f, (int)LX_T82);
        X_T83[gout_idx] = LX_T83;
      }
    }
  }
}
