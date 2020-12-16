#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 32 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 10, 10, 1024 }
// Out stride: { 102400, 10240, 1024, 1 }
// Elementwise input X_T599 shape: fp32(1, 10, 10, 1024):(102400, 10240, 1024, 1):400 KiB
// Elementwise input X_T603 shape: fp32(1024):(1):4 KiB
// Elementwise input X_I_230 shape: fp32(1024):(1):4 KiB
// Elementwise input X_T592 shape: fp32(1, 10, 10, 1024):(102400, 10240, 1024, 1):400 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T604 = div(X_T599, X_T603)
// Elementwise op: [[pid(Add, Switch)]] X_T605 = add(X_T604, X_I_230)
// Elementwise op: [[pid(Add)]] X_T606 = add(X_T592, X_T605)
// Tile size: { 1, 2, 10, 32 }
// Contraction output var shape: fp32(1, 10, 10, 1024):(102400, 10240, 1024, 1):400 KiB
// Computed true ops: 307200
// Computed work groups: 160
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 3072
// Computed mem read: 320
// Computed mem write: 2560
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1280, 32, 1
__kernel void kernel_c28_sdk_183(__global float* restrict  X_T606, __global const float* restrict  X_T599, __global const float* restrict  X_T603, __global const float* restrict  X_I_230, __global const float* restrict  X_T592)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i2_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 3; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 2) || (i3_tid < 2));
    if (i3_cond)
    {
      int i3 = ((4 * i3_lid) + i3_tid);
      int gout_idx = (((10240 * (i2_gid + i2_tid)) + (1024 * i3)) + (i4_gid + i4_tid));
      float LX_T599 = X_T599[gout_idx];
      float LX_T603 = X_T603[(i4_gid + i4_tid)];
      float LX_I_230 = X_I_230[(i4_gid + i4_tid)];
      float LX_T592 = X_T592[gout_idx];
      float LX_T604 = (LX_T599 / LX_T603);
      float LX_T605 = (LX_T604 + LX_I_230);
      float LX_T606 = (LX_T592 + LX_T605);
      X_T606[gout_idx] = LX_T606;
    }
  }
}
