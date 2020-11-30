; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -instcombine -S | FileCheck %s

; a & (a ^ b) --> a & ~b

define i32 @and_xor_common_op(i32 %pa, i32 %pb) {
; CHECK-LABEL: @and_xor_common_op(
; CHECK-NEXT:    [[A:%.*]] = udiv i32 42, [[PA:%.*]]
; CHECK-NEXT:    [[B:%.*]] = udiv i32 43, [[PB:%.*]]
; CHECK-NEXT:    [[TMP1:%.*]] = xor i32 [[B]], -1
; CHECK-NEXT:    [[R:%.*]] = and i32 [[A]], [[TMP1]]
; CHECK-NEXT:    ret i32 [[R]]
;
  %a = udiv i32 42, %pa ; thwart complexity-based canonicalization
  %b = udiv i32 43, %pb ; thwart complexity-based canonicalization
  %xor = xor i32 %a, %b
  %r = and i32 %a, %xor
  ret i32 %r
}

; a & (b ^ a) --> a & ~b

define i32 @and_xor_common_op_commute1(i32 %pa, i32 %pb) {
; CHECK-LABEL: @and_xor_common_op_commute1(
; CHECK-NEXT:    [[A:%.*]] = udiv i32 42, [[PA:%.*]]
; CHECK-NEXT:    [[B:%.*]] = udiv i32 43, [[PB:%.*]]
; CHECK-NEXT:    [[TMP1:%.*]] = xor i32 [[B]], -1
; CHECK-NEXT:    [[R:%.*]] = and i32 [[A]], [[TMP1]]
; CHECK-NEXT:    ret i32 [[R]]
;
  %a = udiv i32 42, %pa ; thwart complexity-based canonicalization
  %b = udiv i32 43, %pb ; thwart complexity-based canonicalization
  %xor = xor i32 %b, %a
  %r = and i32 %a, %xor
  ret i32 %r
}

; (b ^ a) & a --> a & ~b

define i32 @and_xor_common_op_commute2(i32 %pa, i32 %pb) {
; CHECK-LABEL: @and_xor_common_op_commute2(
; CHECK-NEXT:    [[A:%.*]] = udiv i32 42, [[PA:%.*]]
; CHECK-NEXT:    [[B:%.*]] = udiv i32 43, [[PB:%.*]]
; CHECK-NEXT:    [[TMP1:%.*]] = xor i32 [[B]], -1
; CHECK-NEXT:    [[R:%.*]] = and i32 [[A]], [[TMP1]]
; CHECK-NEXT:    ret i32 [[R]]
;
  %a = udiv i32 42, %pa ; thwart complexity-based canonicalization
  %b = udiv i32 43, %pb ; thwart complexity-based canonicalization
  %xor = xor i32 %b, %a
  %r = and i32 %xor, %a
  ret i32 %r
}

; (a ^ b) & a --> a & ~b

define <2 x i32> @and_xor_common_op_commute3(<2 x i32> %pa, <2 x i32> %pb) {
; CHECK-LABEL: @and_xor_common_op_commute3(
; CHECK-NEXT:    [[A:%.*]] = udiv <2 x i32> <i32 42, i32 43>, [[PA:%.*]]
; CHECK-NEXT:    [[B:%.*]] = udiv <2 x i32> <i32 43, i32 42>, [[PB:%.*]]
; CHECK-NEXT:    [[TMP1:%.*]] = xor <2 x i32> [[B]], <i32 -1, i32 -1>
; CHECK-NEXT:    [[R:%.*]] = and <2 x i32> [[A]], [[TMP1]]
; CHECK-NEXT:    ret <2 x i32> [[R]]
;
  %a = udiv <2 x i32> <i32 42, i32 43>, %pa ; thwart complexity-based canonicalization
  %b = udiv <2 x i32> <i32 43, i32 42>, %pb ; thwart complexity-based canonicalization
  %xor = xor <2 x i32> %a, %b
  %r = and <2 x i32> %xor, %a
  ret <2 x i32> %r
}

; It's ok to match a common constant.
; The xor should be a 'not' op (-1 constant).

define <4 x i32> @and_xor_common_op_constant(<4 x i32> %A) {
; CHECK-LABEL: @and_xor_common_op_constant(
; CHECK-NEXT:    [[TMP1:%.*]] = xor <4 x i32> [[A:%.*]], <i32 -1, i32 -1, i32 -1, i32 -1>
; CHECK-NEXT:    [[TMP2:%.*]] = and <4 x i32> [[TMP1]], <i32 1, i32 2, i32 3, i32 4>
; CHECK-NEXT:    ret <4 x i32> [[TMP2]]
;
  %1 = xor <4 x i32> %A, <i32 1, i32 2, i32 3, i32 4>
  %2 = and <4 x i32> <i32 1, i32 2, i32 3, i32 4>, %1
  ret <4 x i32> %2
}

; a & (a ^ ~b) --> a & b

define i32 @and_xor_not_common_op(i32 %a, i32 %b) {
; CHECK-LABEL: @and_xor_not_common_op(
; CHECK-NEXT:    [[T4:%.*]] = and i32 [[A:%.*]], [[B:%.*]]
; CHECK-NEXT:    ret i32 [[T4]]
;
  %b2 = xor i32 %b, -1
  %t2 = xor i32 %a, %b2
  %t4 = and i32 %t2, %a
  ret i32 %t4
}

; rdar://10770603
; (x & y) | (x ^ y) -> x | y

define i64 @or(i64 %x, i64 %y) {
; CHECK-LABEL: @or(
; CHECK-NEXT:    [[TMP1:%.*]] = or i64 [[Y:%.*]], [[X:%.*]]
; CHECK-NEXT:    ret i64 [[TMP1]]
;
  %1 = and i64 %y, %x
  %2 = xor i64 %y, %x
  %3 = add i64 %1, %2
  ret i64 %3
}

; (x & y) + (x ^ y) -> x | y

define i64 @or2(i64 %x, i64 %y) {
; CHECK-LABEL: @or2(
; CHECK-NEXT:    [[TMP1:%.*]] = or i64 [[Y:%.*]], [[X:%.*]]
; CHECK-NEXT:    ret i64 [[TMP1]]
;
  %1 = and i64 %y, %x
  %2 = xor i64 %y, %x
  %3 = or i64 %1, %2
  ret i64 %3
}

; PR37098 - https://bugs.llvm.org/show_bug.cgi?id=37098
; Reassociate bitwise logic to eliminate a shift.
; There are 4 commuted * 3 shift ops * 3 logic ops = 36 potential variations of this fold.
; Mix the commutation options to provide coverage using less tests.

define i8 @and_shl(i8 %x, i8 %y, i8 %z, i8 %shamt) {
; CHECK-LABEL: @and_shl(
; CHECK-NEXT:    [[SX:%.*]] = shl i8 [[X:%.*]], [[SHAMT:%.*]]
; CHECK-NEXT:    [[SY:%.*]] = shl i8 [[Y:%.*]], [[SHAMT]]
; CHECK-NEXT:    [[A:%.*]] = and i8 [[SX]], [[Z:%.*]]
; CHECK-NEXT:    [[R:%.*]] = and i8 [[SY]], [[A]]
; CHECK-NEXT:    ret i8 [[R]]
;
  %sx = shl i8 %x, %shamt
  %sy = shl i8 %y, %shamt
  %a = and i8 %sx, %z
  %r = and i8 %sy, %a
  ret i8 %r
}

define i8 @or_shl(i8 %x, i8 %y, i8 %z, i8 %shamt) {
; CHECK-LABEL: @or_shl(
; CHECK-NEXT:    [[SX:%.*]] = shl i8 [[X:%.*]], [[SHAMT:%.*]]
; CHECK-NEXT:    [[SY:%.*]] = shl i8 [[Y:%.*]], [[SHAMT]]
; CHECK-NEXT:    [[A:%.*]] = or i8 [[SX]], [[Z:%.*]]
; CHECK-NEXT:    [[R:%.*]] = or i8 [[A]], [[SY]]
; CHECK-NEXT:    ret i8 [[R]]
;
  %sx = shl i8 %x, %shamt
  %sy = shl i8 %y, %shamt
  %a = or i8 %sx, %z
  %r = or i8 %a, %sy
  ret i8 %r
}

define i8 @xor_shl(i8 %x, i8 %y, i8 %zarg, i8 %shamt) {
; CHECK-LABEL: @xor_shl(
; CHECK-NEXT:    [[Z:%.*]] = sdiv i8 42, [[ZARG:%.*]]
; CHECK-NEXT:    [[SX:%.*]] = shl i8 [[X:%.*]], [[SHAMT:%.*]]
; CHECK-NEXT:    [[SY:%.*]] = shl i8 [[Y:%.*]], [[SHAMT]]
; CHECK-NEXT:    [[A:%.*]] = xor i8 [[Z]], [[SX]]
; CHECK-NEXT:    [[R:%.*]] = xor i8 [[A]], [[SY]]
; CHECK-NEXT:    ret i8 [[R]]
;
  %z = sdiv i8 42, %zarg ; thwart complexity-based canonicalization
  %sx = shl i8 %x, %shamt
  %sy = shl i8 %y, %shamt
  %a = xor i8 %z, %sx
  %r = xor i8 %a, %sy
  ret i8 %r
}

define i8 @and_lshr(i8 %x, i8 %y, i8 %zarg, i8 %shamt) {
; CHECK-LABEL: @and_lshr(
; CHECK-NEXT:    [[Z:%.*]] = sdiv i8 42, [[ZARG:%.*]]
; CHECK-NEXT:    [[SX:%.*]] = lshr i8 [[X:%.*]], [[SHAMT:%.*]]
; CHECK-NEXT:    [[SY:%.*]] = lshr i8 [[Y:%.*]], [[SHAMT]]
; CHECK-NEXT:    [[A:%.*]] = and i8 [[Z]], [[SX]]
; CHECK-NEXT:    [[R:%.*]] = and i8 [[SY]], [[A]]
; CHECK-NEXT:    ret i8 [[R]]
;
  %z = sdiv i8 42, %zarg ; thwart complexity-based canonicalization
  %sx = lshr i8 %x, %shamt
  %sy = lshr i8 %y, %shamt
  %a = and i8 %z, %sx
  %r = and i8 %sy, %a
  ret i8 %r
}

define i8 @or_lshr(i8 %x, i8 %y, i8 %z, i8 %shamt) {
; CHECK-LABEL: @or_lshr(
; CHECK-NEXT:    [[SX:%.*]] = lshr i8 [[X:%.*]], [[SHAMT:%.*]]
; CHECK-NEXT:    [[SY:%.*]] = lshr i8 [[Y:%.*]], [[SHAMT]]
; CHECK-NEXT:    [[A:%.*]] = or i8 [[SX]], [[Z:%.*]]
; CHECK-NEXT:    [[R:%.*]] = or i8 [[SY]], [[A]]
; CHECK-NEXT:    ret i8 [[R]]
;
  %sx = lshr i8 %x, %shamt
  %sy = lshr i8 %y, %shamt
  %a = or i8 %sx, %z
  %r = or i8 %sy, %a
  ret i8 %r
}

define i8 @xor_lshr(i8 %x, i8 %y, i8 %z, i8 %shamt) {
; CHECK-LABEL: @xor_lshr(
; CHECK-NEXT:    [[SX:%.*]] = lshr i8 [[X:%.*]], [[SHAMT:%.*]]
; CHECK-NEXT:    [[SY:%.*]] = lshr i8 [[Y:%.*]], [[SHAMT]]
; CHECK-NEXT:    [[A:%.*]] = xor i8 [[SX]], [[Z:%.*]]
; CHECK-NEXT:    [[R:%.*]] = xor i8 [[A]], [[SY]]
; CHECK-NEXT:    ret i8 [[R]]
;
  %sx = lshr i8 %x, %shamt
  %sy = lshr i8 %y, %shamt
  %a = xor i8 %sx, %z
  %r = xor i8 %a, %sy
  ret i8 %r
}

define i8 @and_ashr(i8 %x, i8 %y, i8 %zarg, i8 %shamt) {
; CHECK-LABEL: @and_ashr(
; CHECK-NEXT:    [[Z:%.*]] = sdiv i8 42, [[ZARG:%.*]]
; CHECK-NEXT:    [[SX:%.*]] = ashr i8 [[X:%.*]], [[SHAMT:%.*]]
; CHECK-NEXT:    [[SY:%.*]] = ashr i8 [[Y:%.*]], [[SHAMT]]
; CHECK-NEXT:    [[A:%.*]] = and i8 [[Z]], [[SX]]
; CHECK-NEXT:    [[R:%.*]] = and i8 [[A]], [[SY]]
; CHECK-NEXT:    ret i8 [[R]]
;
  %z = sdiv i8 42, %zarg ; thwart complexity-based canonicalization
  %sx = ashr i8 %x, %shamt
  %sy = ashr i8 %y, %shamt
  %a = and i8 %z, %sx
  %r = and i8 %a, %sy
  ret i8 %r
}

define i8 @or_ashr(i8 %x, i8 %y, i8 %zarg, i8 %shamt) {
; CHECK-LABEL: @or_ashr(
; CHECK-NEXT:    [[Z:%.*]] = sdiv i8 42, [[ZARG:%.*]]
; CHECK-NEXT:    [[SX:%.*]] = ashr i8 [[X:%.*]], [[SHAMT:%.*]]
; CHECK-NEXT:    [[SY:%.*]] = ashr i8 [[Y:%.*]], [[SHAMT]]
; CHECK-NEXT:    [[A:%.*]] = or i8 [[Z]], [[SX]]
; CHECK-NEXT:    [[R:%.*]] = or i8 [[SY]], [[A]]
; CHECK-NEXT:    ret i8 [[R]]
;
  %z = sdiv i8 42, %zarg ; thwart complexity-based canonicalization
  %sx = ashr i8 %x, %shamt
  %sy = ashr i8 %y, %shamt
  %a = or i8 %z, %sx
  %r = or i8 %sy, %a
  ret i8 %r
}

define <2 x i8> @xor_ashr(<2 x i8> %x, <2 x i8> %y, <2 x i8> %z, <2 x i8> %shamt) {
; CHECK-LABEL: @xor_ashr(
; CHECK-NEXT:    [[SX:%.*]] = ashr <2 x i8> [[X:%.*]], [[SHAMT:%.*]]
; CHECK-NEXT:    [[SY:%.*]] = ashr <2 x i8> [[Y:%.*]], [[SHAMT]]
; CHECK-NEXT:    [[A:%.*]] = xor <2 x i8> [[SX]], [[Z:%.*]]
; CHECK-NEXT:    [[R:%.*]] = xor <2 x i8> [[A]], [[SY]]
; CHECK-NEXT:    ret <2 x i8> [[R]]
;
  %sx = ashr <2 x i8> %x, %shamt
  %sy = ashr <2 x i8> %y, %shamt
  %a = xor <2 x i8> %sx, %z
  %r = xor <2 x i8> %a, %sy
  ret <2 x i8> %r
}

; Negative test - different logic ops

define i8 @or_and_shl(i8 %x, i8 %y, i8 %z, i8 %shamt) {
; CHECK-LABEL: @or_and_shl(
; CHECK-NEXT:    [[SX:%.*]] = shl i8 [[X:%.*]], [[SHAMT:%.*]]
; CHECK-NEXT:    [[SY:%.*]] = shl i8 [[Y:%.*]], [[SHAMT]]
; CHECK-NEXT:    [[A:%.*]] = or i8 [[SX]], [[Z:%.*]]
; CHECK-NEXT:    [[R:%.*]] = and i8 [[SY]], [[A]]
; CHECK-NEXT:    ret i8 [[R]]
;
  %sx = shl i8 %x, %shamt
  %sy = shl i8 %y, %shamt
  %a = or i8 %sx, %z
  %r = and i8 %sy, %a
  ret i8 %r
}

; Negative test - different shift ops

define i8 @or_lshr_shl(i8 %x, i8 %y, i8 %z, i8 %shamt) {
; CHECK-LABEL: @or_lshr_shl(
; CHECK-NEXT:    [[SX:%.*]] = lshr i8 [[X:%.*]], [[SHAMT:%.*]]
; CHECK-NEXT:    [[SY:%.*]] = shl i8 [[Y:%.*]], [[SHAMT]]
; CHECK-NEXT:    [[A:%.*]] = or i8 [[SX]], [[Z:%.*]]
; CHECK-NEXT:    [[R:%.*]] = or i8 [[A]], [[SY]]
; CHECK-NEXT:    ret i8 [[R]]
;
  %sx = lshr i8 %x, %shamt
  %sy = shl i8 %y, %shamt
  %a = or i8 %sx, %z
  %r = or i8 %a, %sy
  ret i8 %r
}

; Negative test - different shift amounts

define i8 @or_lshr_shamt2(i8 %x, i8 %y, i8 %z, i8 %shamt) {
; CHECK-LABEL: @or_lshr_shamt2(
; CHECK-NEXT:    [[SX:%.*]] = lshr i8 [[X:%.*]], 5
; CHECK-NEXT:    [[SY:%.*]] = lshr i8 [[Y:%.*]], [[SHAMT:%.*]]
; CHECK-NEXT:    [[A:%.*]] = or i8 [[SX]], [[Z:%.*]]
; CHECK-NEXT:    [[R:%.*]] = or i8 [[SY]], [[A]]
; CHECK-NEXT:    ret i8 [[R]]
;
  %sx = lshr i8 %x, 5
  %sy = lshr i8 %y, %shamt
  %a = or i8 %sx, %z
  %r = or i8 %sy, %a
  ret i8 %r
}

; Negative test - multi-use

define i8 @xor_lshr_multiuse(i8 %x, i8 %y, i8 %z, i8 %shamt) {
; CHECK-LABEL: @xor_lshr_multiuse(
; CHECK-NEXT:    [[SX:%.*]] = lshr i8 [[X:%.*]], [[SHAMT:%.*]]
; CHECK-NEXT:    [[SY:%.*]] = lshr i8 [[Y:%.*]], [[SHAMT]]
; CHECK-NEXT:    [[A:%.*]] = xor i8 [[SX]], [[Z:%.*]]
; CHECK-NEXT:    [[R:%.*]] = xor i8 [[A]], [[SY]]
; CHECK-NEXT:    [[R2:%.*]] = sdiv i8 [[A]], [[R]]
; CHECK-NEXT:    ret i8 [[R2]]
;
  %sx = lshr i8 %x, %shamt
  %sy = lshr i8 %y, %shamt
  %a = xor i8 %sx, %z
  %r = xor i8 %a, %sy
  %r2 = sdiv i8 %a, %r
  ret i8 %r2
}

