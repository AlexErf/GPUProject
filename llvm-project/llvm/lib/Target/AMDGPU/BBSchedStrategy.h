//===-- BBSchedStrategy.h - BB Scheduler Strategy -*- C++ -*-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_BBSCHEDSTRATEGY_H
#define LLVM_LIB_TARGET_AMDGPU_BBSCHEDSTRATEGY_H

#include "GCNRegPressure.h"
#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {

class SIMachineFunctionInfo;
class SIRegisterInfo;
class GCNSubtarget;

/// This is a minimal scheduler strategy.  The main difference between this
/// and the GenericScheduler is that GCNSchedStrategy uses different
/// heuristics to determine excess/critical pressure sets.  Its goal is to
/// maximize kernel occupancy (i.e. maximum number of waves per simd).
class GCNSchedStrategy final : public GenericScheduler {
  friend class GCNScheduleDAGMILive;

  SUnit *pickNodeBidirectional(bool &IsTopNode);

  void pickNodeFromQueue(SchedBoundary &Zone, const CandPolicy &ZonePolicy,
                         const RegPressureTracker &RPTracker,
                         SchedCandidate &Cand);

  void initCandidate(SchedCandidate &Cand, SUnit *SU,
                     bool AtTop, const RegPressureTracker &RPTracker,
                     const SIRegisterInfo *SRI,
                     unsigned SGPRPressure, unsigned VGPRPressure);

  std::vector<unsigned> Pressure;
  std::vector<unsigned> MaxPressure;

  unsigned SGPRExcessLimit;
  unsigned VGPRExcessLimit;
  unsigned SGPRCriticalLimit;
  unsigned VGPRCriticalLimit;

  unsigned TargetOccupancy;

  MachineFunction *MF;

  std::map<SUnit*, int> estart;
  std::map<SUnit*, int> lstart;

protected:  

  void GCNSchedStrategy::computeEstart(SmallVector<SUnit*, 8> topRoots);

  void GCNSchedStrategy::computeLstart(SmallVector<SUnit*, 8> bottomRoots, int maxEstart);

  bool GCNSchedStrategy::cmp(pair<SUnit*, int> &a, pair<SUnit*, int> &b);

  unsigned GCNSchedStrategy::computeDLB(std::map<int, SUnit*> scheduleSoFar);

  bool GCNSchedStrategy::checkNode(std::map<SUnit*, int> scheduleSoFar, unsigned targetLength, unsigned targetAPRP, 
                                  unsigned enumBestAPRP, bool isOccupancyPass);
public:
  BBSchedStrategy(const MachineSchedContext *C);

  SUnit *pickNode(bool &IsTopNode) override;

  void initialize(ScheduleDAGMI *DAG) override;

  void setTargetOccupancy(unsigned Occ) { TargetOccupancy = Occ; }
};

class GCNScheduleDAGMILive final : public ScheduleDAGMILive {

  enum : unsigned {
    Collect,
    InitialSchedule,
    UnclusteredReschedule,
    ClusteredLowOccupancyReschedule,
    LastStage = ClusteredLowOccupancyReschedule
  };

  const GCNSubtarget &ST;

  SIMachineFunctionInfo &MFI;

  // Occupancy target at the beginning of function scheduling cycle.
  unsigned StartingOccupancy;

  // Minimal real occupancy recorder for the function.
  unsigned MinOccupancy;

  // Scheduling stage number.
  unsigned Stage;

  // Current region index.
  size_t RegionIdx;

  // Vector of regions recorder for later rescheduling
  SmallVector<std::pair<MachineBasicBlock::iterator,
                        MachineBasicBlock::iterator>, 32> Regions;

  // Records if a region is not yet scheduled, or schedule has been reverted,
  // or we generally desire to reschedule it.
  BitVector RescheduleRegions;

  // Region live-in cache.
  SmallVector<GCNRPTracker::LiveRegSet, 32> LiveIns;

  // Region pressure cache.
  SmallVector<GCNRegPressure, 32> Pressure;

  // Temporary basic block live-in cache.
  DenseMap<const MachineBasicBlock*, GCNRPTracker::LiveRegSet> MBBLiveIns;

  DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet> BBLiveInMap;
  DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet> getBBLiveInMap() const;

  // Return current region pressure.
  GCNRegPressure getRealRegPressure() const;

  // Compute and cache live-ins and pressure for all regions in block.
  void computeBlockPressure(const MachineBasicBlock *MBB);

  void GCNScheduleDAGMILive::occupancyPass(std::vector<SUnit* > topRoots, int targetPressure);
  void GCNScheduleDAGMILive::ilpPass(std::vector<SUnit* > topRoots, std::vector<SUnit* > scheduleInst, int targetPressure);
  int GCNScheduleDAGMILive::enumerate(SmallVector<SUnit*, 8> TopRoots, SmallVector<SUnit*, 8> BottomRoots,
                                      unsigned targetLength, unsigned targetAPRP, bool isOccupanyPass);
  void GCNScheduleDAGMILive::scheduleInst(MachineInstr* MI);

public:
  GCNScheduleDAGMILive(MachineSchedContext *C,
                       std::unique_ptr<MachineSchedStrategy> S);

  void schedule() override;

  void finalizeSchedule() override;
};

} // End namespace llvm

#endif // GCNSCHEDSTRATEGY_H
