#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 384 }
// Out stride: { 1 }
// Elementwise input X_I_172 shape: fp32(384):(1):1.5 KiB
// Elementwise op: [[pid(Add)]] X_T436 = add(X_T43, X_I_172)
// Elementwise op: X_T437 = cmp_lt(X_T436, X_T20)
// Elementwise op: [[pid(Sqrt)]] X_T438 = cond(X_T437, X_T20, X_T436)
// Elementwise op: [[pid(Sqrt)]] X_T439 = sqrt(X_T438)
// Tile size: { 256 }
// Contraction output var shape: fp32(384):(1):1.5 KiB
// Computed true ops: 1536
// Computed work groups: 2
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 512, 1, 1
__kernel void kernel_c68_sdk_139(__global float* restrict  X_T439, __global const float* restrict  X_I_172)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 256) || (i1_tid < 128));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_172 = X_I_172[gout_idx];
    float LX_T436 = (1.0009999641624745e-5f + LX_I_172);
    int LX_T437 = (LX_T436 < (float)0);
    float LX_T438 = select((float)LX_T436, (float)0, (int)LX_T437);
    float LX_T439 = native_sqrt(LX_T438);
    X_T439[gout_idx] = LX_T439;
  }
}
