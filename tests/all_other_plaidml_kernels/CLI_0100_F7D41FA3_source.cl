#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i1 }
// Ranges: { 96 }
// Out stride: { 1 }
// Elementwise input X_I_38 shape: fp32(96):(1):384 bytes
// Elementwise op: [[pid(Add)]] X_T87 = add(X_T33, X_I_38)
// Elementwise op: X_T88 = cmp_lt(X_T87, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T89 = cond(X_T88, X_T4, X_T87)
// Elementwise op: [[pid(Sqrt)]] X_T90 = sqrt(X_T89)
// Tile size: { 96 }
// Contraction output var shape: fp32(96):(1):384 bytes
// Computed true ops: 384
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 12
// Computed mem write: 384
// Computed operations: 96
// Computed rollups: 0
// Computed threads used: 128
// lwork = 128, 1, 1
// gwork = 128, 1, 1
__kernel void kernel_c51_sdk_18(__global float* restrict  X_T90, __global const float* restrict  X_I_38)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 128);
  int i1_cond = (i1_tid < 96);
  if (i1_cond)
  {
    float LX_I_38 = X_I_38[i1_tid];
    float LX_T87 = (0.0010000000474974513f + LX_I_38);
    int LX_T88 = (LX_T87 < (float)0);
    float LX_T89 = select((float)LX_T87, (float)0, (int)LX_T88);
    float LX_T90 = native_sqrt(LX_T89);
    X_T90[i1_tid] = LX_T90;
  }
}
