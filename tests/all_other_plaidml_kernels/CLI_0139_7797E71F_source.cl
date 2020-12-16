#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 224 }
// Out stride: { 1 }
// Elementwise input X_I_81 shape: fp32(224):(1):896 bytes
// Elementwise op: [[pid(Add)]] X_T190 = add(X_T43, X_I_81)
// Elementwise op: X_T191 = cmp_lt(X_T190, X_T20)
// Elementwise op: [[pid(Sqrt)]] X_T192 = cond(X_T191, X_T20, X_T190)
// Elementwise op: [[pid(Sqrt)]] X_T193 = sqrt(X_T192)
// Tile size: { 224 }
// Contraction output var shape: fp32(224):(1):896 bytes
// Computed true ops: 896
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 28
// Computed mem write: 896
// Computed operations: 224
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 256, 1, 1
__kernel void kernel_c68_sdk_52(__global float* restrict  X_T193, __global const float* restrict  X_I_81)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  int i1_cond = (i1_tid < 224);
  if (i1_cond)
  {
    float LX_I_81 = X_I_81[i1_tid];
    float LX_T190 = (1.0009999641624745e-5f + LX_I_81);
    int LX_T191 = (LX_T190 < (float)0);
    float LX_T192 = select((float)LX_T190, (float)0, (int)LX_T191);
    float LX_T193 = native_sqrt(LX_T192);
    X_T193[i1_tid] = LX_T193;
  }
}
