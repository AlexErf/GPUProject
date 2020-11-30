; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -loop-reduce -S %s | FileCheck %s

; Make sure SCEVExpander does not crash and introduce unnecessary LCSSA PHI nodes.

define void @schedule_block() {
; CHECK-LABEL: @schedule_block(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    switch i16 undef, label [[IF_END156_I:%.*]] [
; CHECK-NEXT:    i16 27, label [[IF_THEN_I:%.*]]
; CHECK-NEXT:    i16 28, label [[IF_THEN_I]]
; CHECK-NEXT:    i16 29, label [[IF_THEN13_I:%.*]]
; CHECK-NEXT:    i16 32, label [[LAND_LHS_TRUE136_I:%.*]]
; CHECK-NEXT:    ]
; CHECK:       if.then.i:
; CHECK-NEXT:    unreachable
; CHECK:       if.then13.i:
; CHECK-NEXT:    unreachable
; CHECK:       land.lhs.true136.i:
; CHECK-NEXT:    unreachable
; CHECK:       if.end156.i:
; CHECK-NEXT:    switch i16 undef, label [[WHILE_END256:%.*]] [
; CHECK-NEXT:    i16 29, label [[IF_THEN210:%.*]]
; CHECK-NEXT:    i16 28, label [[IF_THEN210]]
; CHECK-NEXT:    i16 27, label [[LAND_LHS_TRUE191:%.*]]
; CHECK-NEXT:    i16 32, label [[IF_END248:%.*]]
; CHECK-NEXT:    ]
; CHECK:       land.lhs.true191:
; CHECK-NEXT:    unreachable
; CHECK:       if.then210:
; CHECK-NEXT:    unreachable
; CHECK:       if.end248:
; CHECK-NEXT:    br label [[FOR_END:%.*]]
; CHECK:       while.end256:
; CHECK-NEXT:    unreachable
; CHECK:       for.end:
; CHECK-NEXT:    br label [[WHILE_BODY1013:%.*]]
; CHECK:       while.body1013:
; CHECK-NEXT:    br label [[FOR_COND_I2472:%.*]]
; CHECK:       for.cond.i2472:
; CHECK-NEXT:    [[I_0_I:%.*]] = phi i32 [ 0, [[WHILE_BODY1013]] ], [ [[TMP2:%.*]], [[FOR_END34_I:%.*]] ]
; CHECK-NEXT:    br i1 false, label [[FOR_COND3_PREHEADER_I:%.*]], label [[IF_END107_I_LOOPEXIT:%.*]]
; CHECK:       for.cond3.preheader.i:
; CHECK-NEXT:    [[TMP0:%.*]] = sext i32 [[I_0_I]] to i64
; CHECK-NEXT:    [[TMP1:%.*]] = add nsw i64 [[TMP0]], 1
; CHECK-NEXT:    br label [[FOR_COND3_I:%.*]]
; CHECK:       for.cond3.i:
; CHECK-NEXT:    [[INDVARS_IV301_I2691:%.*]] = phi i64 [ [[INDVARS_IV_NEXT302_I:%.*]], [[FOR_BODY5_I:%.*]] ], [ [[TMP1]], [[FOR_COND3_PREHEADER_I]] ]
; CHECK-NEXT:    [[INDVARS_IV_NEXT302_I]] = add nsw i64 [[INDVARS_IV301_I2691]], 1
; CHECK-NEXT:    br label [[FOR_BODY5_I]]
; CHECK:       for.body5.i:
; CHECK-NEXT:    br i1 false, label [[FOR_COND3_I]], label [[FOR_BODY5_I_FOR_END_I2475_LOOPEXIT_CRIT_EDGE:%.*]]
; CHECK:       for.body5.i.for.end.i2475.loopexit_crit_edge:
; CHECK-NEXT:    [[TMP2]] = trunc i64 [[INDVARS_IV_NEXT302_I]] to i32
; CHECK-NEXT:    br label [[FOR_END34_I]]
; CHECK:       for.end34.i:
; CHECK-NEXT:    br i1 false, label [[FOR_COND_I2472]], label [[IF_ELSE_I2488:%.*]]
; CHECK:       if.else.i2488:
; CHECK-NEXT:    br i1 undef, label [[IF_END107_I:%.*]], label [[FOR_BODY45_PREHEADER_I:%.*]]
; CHECK:       for.body45.preheader.i:
; CHECK-NEXT:    [[TMP3:%.*]] = sext i32 [[I_0_I]] to i64
; CHECK-NEXT:    unreachable
; CHECK:       if.end107.i.loopexit:
; CHECK-NEXT:    br label [[IF_END107_I]]
; CHECK:       if.end107.i:
; CHECK-NEXT:    unreachable
;
entry:
  switch i16 undef, label %if.end156.i [
  i16 27, label %if.then.i
  i16 28, label %if.then.i
  i16 29, label %if.then13.i
  i16 32, label %land.lhs.true136.i
  ]

if.then.i:                                        ; preds = %entry, %entry
  unreachable

if.then13.i:                                      ; preds = %entry
  unreachable

land.lhs.true136.i:                               ; preds = %entry
  unreachable

if.end156.i:                                      ; preds = %entry
  switch i16 undef, label %while.end256 [
  i16 29, label %if.then210
  i16 28, label %if.then210
  i16 27, label %land.lhs.true191
  i16 32, label %if.end248
  ]

land.lhs.true191:                                 ; preds = %if.end156.i
  unreachable

if.then210:                                       ; preds = %if.end156.i, %if.end156.i
  unreachable

if.end248:                                        ; preds = %if.end156.i
  br label %for.end

while.end256:                                     ; preds = %if.end156.i
  unreachable

for.end:                                          ; preds = %if.end248
  br label %while.body1013

while.body1013:                                   ; preds = %for.end
  br label %for.cond.i2472

for.cond.i2472:                                   ; preds = %for.end34.i, %while.body1013
  %i.0.i = phi i32 [ 0, %while.body1013 ], [ %2, %for.end34.i ]
  br i1 undef, label %for.cond3.preheader.i, label %if.end107.i

for.cond3.preheader.i:                            ; preds = %for.cond.i2472
  %0 = sext i32 %i.0.i to i64
  %1 = add nsw i64 %0, 1
  br label %for.cond3.i

for.cond3.i:                                      ; preds = %for.body5.i, %for.cond3.preheader.i
  %indvars.iv301.i2691 = phi i64 [ %indvars.iv.next302.i, %for.body5.i ], [ %1, %for.cond3.preheader.i ]
  %indvars.iv.next302.i = add nsw i64 %indvars.iv301.i2691, 1
  br label %for.body5.i

for.body5.i:                                      ; preds = %for.cond3.i
  br i1 undef, label %for.cond3.i, label %for.body5.i.for.end.i2475.loopexit_crit_edge

for.body5.i.for.end.i2475.loopexit_crit_edge:     ; preds = %for.body5.i
  %2 = trunc i64 %indvars.iv.next302.i to i32
  br label %for.end34.i

for.end34.i:                                      ; preds = %for.body5.i.for.end.i2475.loopexit_crit_edge
  br i1 undef, label %for.cond.i2472, label %if.else.i2488

if.else.i2488:                                    ; preds = %for.end34.i
  br i1 undef, label %if.end107.i, label %for.body45.preheader.i

for.body45.preheader.i:                           ; preds = %if.else.i2488
  %3 = sext i32 %i.0.i to i64
  unreachable

if.end107.i:                                      ; preds = %if.else.i2488, %for.cond.i2472
  unreachable
}
