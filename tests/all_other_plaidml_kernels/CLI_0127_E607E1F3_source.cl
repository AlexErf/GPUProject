#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 160 }
// Out stride: { 1 }
// Elementwise input X_I_61 shape: fp32(160):(1):640 bytes
// Elementwise op: [[pid(Add)]] X_T140 = add(X_T43, X_I_61)
// Elementwise op: X_T141 = cmp_lt(X_T140, X_T20)
// Elementwise op: [[pid(Sqrt)]] X_T142 = cond(X_T141, X_T20, X_T140)
// Elementwise op: [[pid(Sqrt)]] X_T143 = sqrt(X_T142)
// Tile size: { 160 }
// Contraction output var shape: fp32(160):(1):640 bytes
// Computed true ops: 640
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 20
// Computed mem write: 640
// Computed operations: 160
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 256, 1, 1
__kernel void kernel_c68_sdk_34(__global float* restrict  X_T143, __global const float* restrict  X_I_61)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  int i1_cond = (i1_tid < 160);
  if (i1_cond)
  {
    float LX_I_61 = X_I_61[i1_tid];
    float LX_T140 = (1.0009999641624745e-5f + LX_I_61);
    int LX_T141 = (LX_T140 < (float)0);
    float LX_T142 = select((float)LX_T140, (float)0, (int)LX_T141);
    float LX_T143 = native_sqrt(LX_T142);
    X_T143[i1_tid] = LX_T143;
  }
}
