#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 480 }
// Out stride: { 1 }
// Elementwise input X_I_202 shape: fp32(480):(1):1.875 KiB
// Elementwise op: [[pid(Add)]] X_T539 = add(X_T71, X_I_202)
// Elementwise op: X_T540 = cmp_lt(X_T539, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T541 = cond(X_T540, X_T36, X_T539)
// Elementwise op: [[pid(Sqrt)]] X_T542 = sqrt(X_T541)
// Tile size: { 256 }
// Contraction output var shape: fp32(480):(1):1.875 KiB
// Computed true ops: 1920
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
__kernel void kernel_c124_sdk_166(__global float* restrict  X_T542, __global const float* restrict  X_I_202)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 256) || (i1_tid < 224));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_202 = X_I_202[gout_idx];
    float LX_T539 = (1.0009999641624745e-5f + LX_I_202);
    int LX_T540 = (LX_T539 < (float)0);
    float LX_T541 = select((float)LX_T539, (float)0, (int)LX_T540);
    float LX_T542 = native_sqrt(LX_T541);
    X_T542[gout_idx] = LX_T542;
  }
}
