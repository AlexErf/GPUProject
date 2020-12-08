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

struct node {
    std::vector<node*> preds;
    std::vector<int> predsLatencies;

    std::vector<node*> succs;
    std::vector<int> succsLatencies;

    /// a helper function to get the succ latency
    int getSuccLatency(node* succ) {
        auto it = find(succs.begin(), succs.end(), succ);

        if (it != succs.end()) {
            int index = it - succs.begin();
            return succsLatencies[index];
        }
        return -1;
    }

    /// a helper function to get the pred latency
    int getPredLatency(node* pred) {
        auto it = find(preds.begin(), preds.end(), pred);

        if (it != preds.end()) {
            int index = it - preds.begin();
            return predsLatencies[index];
        }
        return -1;
    }
  };

/// This is a minimal scheduler strategy.  The main difference between this
/// and the GenericScheduler is that GCNSchedStrategy uses different
/// heuristics to determine excess/critical pressure sets.  Its goal is to
/// maximize kernel occupancy (i.e. maximum number of waves per simd).
class GCNMaxOccupancySchedStrategy final : public GenericScheduler {
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

public:
  GCNMaxOccupancySchedStrategy(const MachineSchedContext *C);

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

  void occupancyPass(SmallVector<SUnit*, 8> topRoots, SmallVector<SUnit*, 8> bottomRoots, unsigned targetPressure);
  void ilpPass(SmallVector<SUnit*, 8> topRoots, std::vector<SUnit* > scheduleInst, int targetPressure);
  unsigned enumerate(SmallVector<SUnit*, 8> TopRoots, SmallVector<SUnit*, 8> BottomRoots,
                                      unsigned targetLength, unsigned targetAPRP, bool isOccupanyPass);
  void scheduleInst(MachineInstr* MI);
  int satisfyLatency(std::vector<SUnit*> schedule);

  void restoreLatencies(SmallVector<llvm::SUnit *, 8> &topRoots, std::map<SUnit*, node*> mapSUnitToNode);

  std::map<SUnit*, int> estart;
  std::map<SUnit*, int> lstart;

  std::map<SUnit*, node*> setLatenciesToOne(SmallVector<llvm::SUnit *, 8> &topRoots);

  void computeEstart(SmallVector<SUnit*, 8> topRoots);

  void computeLstart(SmallVector<SUnit*, 8> bottomRoots, int maxEstart);

  unsigned computeDLB(std::map<int, SUnit*> scheduleSoFar);


  bool checkNode(SUnit* node, const std::map<int, SUnit*>& scheduleSoFar, const std::vector<SUnit*>& currentScheduledInstructions, unsigned targetLength, unsigned targetAPRP, 
                                unsigned enumBestAPRP, bool isOccupancyPass);

GCNDownwardRPTracker RPTracker;
bool haveBacktracked; // flag that tells us we need to reset the RPTracker
public:
  GCNScheduleDAGMILive(MachineSchedContext *C,
                       std::unique_ptr<MachineSchedStrategy> S);

  void schedule() override;

  void finalizeSchedule() override;
};

} // End namespace llvm

#endif // BBSCHEDSTRATEGY_H
