//===-- GCNSchedStrategy.cpp - GCN Scheduler Strategy ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This contains a MachineSchedStrategy implementation for maximizing wave
/// occupancy on GCN hardware.
//===----------------------------------------------------------------------===//

#include "BBSchedStrategy.h"
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/Support/MathExtras.h"
#include <queue>
#include <stack>
#include <map>
#include <deque>
#include <algorithm>
#include <unordered_map>


#define DEBUG_TYPE "machine-scheduler"

using namespace llvm;

GCNMaxOccupancySchedStrategy::GCNMaxOccupancySchedStrategy(
    const MachineSchedContext *C) :
    GenericScheduler(C), TargetOccupancy(0), MF(nullptr) { }

void GCNMaxOccupancySchedStrategy::initialize(ScheduleDAGMI *DAG) {
  GenericScheduler::initialize(DAG);

  const SIRegisterInfo *SRI = static_cast<const SIRegisterInfo*>(TRI);

  MF = &DAG->MF;

  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();

  // FIXME: This is also necessary, because some passes that run after
  // scheduling and before regalloc increase register pressure.
  const int ErrorMargin = 3;

  SGPRExcessLimit = Context->RegClassInfo
    ->getNumAllocatableRegs(&AMDGPU::SGPR_32RegClass) - ErrorMargin;
  VGPRExcessLimit = Context->RegClassInfo
    ->getNumAllocatableRegs(&AMDGPU::VGPR_32RegClass) - ErrorMargin;
  if (TargetOccupancy) {
    SGPRCriticalLimit = ST.getMaxNumSGPRs(TargetOccupancy, true);
    VGPRCriticalLimit = ST.getMaxNumVGPRs(TargetOccupancy);
  } else {
    SGPRCriticalLimit = SRI->getRegPressureSetLimit(DAG->MF,
        AMDGPU::RegisterPressureSets::SReg_32);
    VGPRCriticalLimit = SRI->getRegPressureSetLimit(DAG->MF,
        AMDGPU::RegisterPressureSets::VGPR_32);
  }

  SGPRCriticalLimit -= ErrorMargin;
  VGPRCriticalLimit -= ErrorMargin;
}

void GCNMaxOccupancySchedStrategy::initCandidate(SchedCandidate &Cand, SUnit *SU,
                                     bool AtTop, const RegPressureTracker &RPTracker,
                                     const SIRegisterInfo *SRI,
                                     unsigned SGPRPressure,
                                     unsigned VGPRPressure) {

  Cand.SU = SU;
  Cand.AtTop = AtTop;

  // getDownwardPressure() and getUpwardPressure() make temporary changes to
  // the tracker, so we need to pass those function a non-const copy.
  RegPressureTracker &TempTracker = const_cast<RegPressureTracker&>(RPTracker);

  Pressure.clear();
  MaxPressure.clear();

  if (AtTop)
    TempTracker.getDownwardPressure(SU->getInstr(), Pressure, MaxPressure);
  else {
    // FIXME: I think for bottom up scheduling, the register pressure is cached
    // and can be retrieved by DAG->getPressureDif(SU).
    TempTracker.getUpwardPressure(SU->getInstr(), Pressure, MaxPressure);
  }

  unsigned NewSGPRPressure = Pressure[AMDGPU::RegisterPressureSets::SReg_32];
  unsigned NewVGPRPressure = Pressure[AMDGPU::RegisterPressureSets::VGPR_32];

  // If two instructions increase the pressure of different register sets
  // by the same amount, the generic scheduler will prefer to schedule the
  // instruction that increases the set with the least amount of registers,
  // which in our case would be SGPRs.  This is rarely what we want, so
  // when we report excess/critical register pressure, we do it either
  // only for VGPRs or only for SGPRs.

  // FIXME: Better heuristics to determine whether to prefer SGPRs or VGPRs.
  const unsigned MaxVGPRPressureInc = 16;
  bool ShouldTrackVGPRs = VGPRPressure + MaxVGPRPressureInc >= VGPRExcessLimit;
  bool ShouldTrackSGPRs = !ShouldTrackVGPRs && SGPRPressure >= SGPRExcessLimit;


  // FIXME: We have to enter REG-EXCESS before we reach the actual threshold
  // to increase the likelihood we don't go over the limits.  We should improve
  // the analysis to look through dependencies to find the path with the least
  // register pressure.

  // We only need to update the RPDelta for instructions that increase register
  // pressure. Instructions that decrease or keep reg pressure the same will be
  // marked as RegExcess in tryCandidate() when they are compared with
  // instructions that increase the register pressure.
  if (ShouldTrackVGPRs && NewVGPRPressure >= VGPRExcessLimit) {
    Cand.RPDelta.Excess = PressureChange(AMDGPU::RegisterPressureSets::VGPR_32);
    Cand.RPDelta.Excess.setUnitInc(NewVGPRPressure - VGPRExcessLimit);
  }

  if (ShouldTrackSGPRs && NewSGPRPressure >= SGPRExcessLimit) {
    Cand.RPDelta.Excess = PressureChange(AMDGPU::RegisterPressureSets::SReg_32);
    Cand.RPDelta.Excess.setUnitInc(NewSGPRPressure - SGPRExcessLimit);
  }

  // Register pressure is considered 'CRITICAL' if it is approaching a value
  // that would reduce the wave occupancy for the execution unit.  When
  // register pressure is 'CRITICAL', increading SGPR and VGPR pressure both
  // has the same cost, so we don't need to prefer one over the other.

  int SGPRDelta = NewSGPRPressure - SGPRCriticalLimit;
  int VGPRDelta = NewVGPRPressure - VGPRCriticalLimit;

  if (SGPRDelta >= 0 || VGPRDelta >= 0) {
    if (SGPRDelta > VGPRDelta) {
      Cand.RPDelta.CriticalMax =
        PressureChange(AMDGPU::RegisterPressureSets::SReg_32);
      Cand.RPDelta.CriticalMax.setUnitInc(SGPRDelta);
    } else {
      Cand.RPDelta.CriticalMax =
        PressureChange(AMDGPU::RegisterPressureSets::VGPR_32);
      Cand.RPDelta.CriticalMax.setUnitInc(VGPRDelta);
    }
  }
}

// This function is mostly cut and pasted from
// GenericScheduler::pickNodeFromQueue()
void GCNMaxOccupancySchedStrategy::pickNodeFromQueue(SchedBoundary &Zone,
                                         const CandPolicy &ZonePolicy,
                                         const RegPressureTracker &RPTracker,
                                         SchedCandidate &Cand) {
  const SIRegisterInfo *SRI = static_cast<const SIRegisterInfo*>(TRI);
  ArrayRef<unsigned> Pressure = RPTracker.getRegSetPressureAtPos();
  unsigned SGPRPressure = Pressure[AMDGPU::RegisterPressureSets::SReg_32];
  unsigned VGPRPressure = Pressure[AMDGPU::RegisterPressureSets::VGPR_32];
  ReadyQueue &Q = Zone.Available;
  for (SUnit *SU : Q) {

    SchedCandidate TryCand(ZonePolicy);
    initCandidate(TryCand, SU, Zone.isTop(), RPTracker, SRI,
                  SGPRPressure, VGPRPressure);
    // Pass SchedBoundary only when comparing nodes from the same boundary.
    SchedBoundary *ZoneArg = Cand.AtTop == TryCand.AtTop ? &Zone : nullptr;
    GenericScheduler::tryCandidate(Cand, TryCand, ZoneArg);
    if (TryCand.Reason != NoCand) {
      // Initialize resource delta if needed in case future heuristics query it.
      if (TryCand.ResDelta == SchedResourceDelta())
        TryCand.initResourceDelta(Zone.DAG, SchedModel);
      Cand.setBest(TryCand);
      LLVM_DEBUG(traceCandidate(Cand));
    }
  }
}

// This function is mostly cut and pasted from
// GenericScheduler::pickNodeBidirectional()
SUnit *GCNMaxOccupancySchedStrategy::pickNodeBidirectional(bool &IsTopNode) {
  // Schedule as far as possible in the direction of no choice. This is most
  // efficient, but also provides the best heuristics for CriticalPSets.
  if (SUnit *SU = Bot.pickOnlyChoice()) {
    IsTopNode = false;
    return SU;
  }
  if (SUnit *SU = Top.pickOnlyChoice()) {
    IsTopNode = true;
    return SU;
  }
  // Set the bottom-up policy based on the state of the current bottom zone and
  // the instructions outside the zone, including the top zone.
  CandPolicy BotPolicy;
  setPolicy(BotPolicy, /*IsPostRA=*/false, Bot, &Top);
  // Set the top-down policy based on the state of the current top zone and
  // the instructions outside the zone, including the bottom zone.
  CandPolicy TopPolicy;
  setPolicy(TopPolicy, /*IsPostRA=*/false, Top, &Bot);

  // See if BotCand is still valid (because we previously scheduled from Top).
  LLVM_DEBUG(dbgs() << "Picking from Bot:\n");
  if (!BotCand.isValid() || BotCand.SU->isScheduled ||
      BotCand.Policy != BotPolicy) {
    BotCand.reset(CandPolicy());
    pickNodeFromQueue(Bot, BotPolicy, DAG->getBotRPTracker(), BotCand);
    assert(BotCand.Reason != NoCand && "failed to find the first candidate");
  } else {
    LLVM_DEBUG(traceCandidate(BotCand));
#ifndef NDEBUG
    if (VerifyScheduling) {
      SchedCandidate TCand;
      TCand.reset(CandPolicy());
      pickNodeFromQueue(Bot, BotPolicy, DAG->getBotRPTracker(), TCand);
      assert(TCand.SU == BotCand.SU &&
             "Last pick result should correspond to re-picking right now");
    }
#endif
  }

  // Check if the top Q has a better candidate.
  LLVM_DEBUG(dbgs() << "Picking from Top:\n");
  if (!TopCand.isValid() || TopCand.SU->isScheduled ||
      TopCand.Policy != TopPolicy) {
    TopCand.reset(CandPolicy());
    pickNodeFromQueue(Top, TopPolicy, DAG->getTopRPTracker(), TopCand);
    assert(TopCand.Reason != NoCand && "failed to find the first candidate");
  } else {
    LLVM_DEBUG(traceCandidate(TopCand));
#ifndef NDEBUG
    if (VerifyScheduling) {
      SchedCandidate TCand;
      TCand.reset(CandPolicy());
      pickNodeFromQueue(Top, TopPolicy, DAG->getTopRPTracker(), TCand);
      assert(TCand.SU == TopCand.SU &&
           "Last pick result should correspond to re-picking right now");
    }
#endif
  }

  // Pick best from BotCand and TopCand.
  LLVM_DEBUG(dbgs() << "Top Cand: "; traceCandidate(TopCand);
             dbgs() << "Bot Cand: "; traceCandidate(BotCand););
  SchedCandidate Cand = BotCand;
  TopCand.Reason = NoCand;
  GenericScheduler::tryCandidate(Cand, TopCand, nullptr);
  if (TopCand.Reason != NoCand) {
    Cand.setBest(TopCand);
  }
  LLVM_DEBUG(dbgs() << "Picking: "; traceCandidate(Cand););

  IsTopNode = Cand.AtTop;
  return Cand.SU;
}

// This function is mostly cut and pasted from
// GenericScheduler::pickNode()
SUnit *GCNMaxOccupancySchedStrategy::pickNode(bool &IsTopNode) {
  if (DAG->top() == DAG->bottom()) {
    assert(Top.Available.empty() && Top.Pending.empty() &&
           Bot.Available.empty() && Bot.Pending.empty() && "ReadyQ garbage");
    return nullptr;
  }
  SUnit *SU;
  do {
    if (RegionPolicy.OnlyTopDown) {
      SU = Top.pickOnlyChoice();
      if (!SU) {
        CandPolicy NoPolicy;
        TopCand.reset(NoPolicy);
        pickNodeFromQueue(Top, NoPolicy, DAG->getTopRPTracker(), TopCand);
        assert(TopCand.Reason != NoCand && "failed to find a candidate");
        SU = TopCand.SU;
      }
      IsTopNode = true;
    } else if (RegionPolicy.OnlyBottomUp) {
      SU = Bot.pickOnlyChoice();
      if (!SU) {
        CandPolicy NoPolicy;
        BotCand.reset(NoPolicy);
        pickNodeFromQueue(Bot, NoPolicy, DAG->getBotRPTracker(), BotCand);
        assert(BotCand.Reason != NoCand && "failed to find a candidate");
        SU = BotCand.SU;
      }
      IsTopNode = false;
    } else {
      SU = pickNodeBidirectional(IsTopNode);
    }
  } while (SU->isScheduled);

  if (SU->isTopReady())
    Top.removeReady(SU);
  if (SU->isBottomReady())
    Bot.removeReady(SU);

  LLVM_DEBUG(dbgs() << "Scheduling SU(" << SU->NodeNum << ") "
                    << *SU->getInstr());
  return SU;
}

GCNScheduleDAGMILive::GCNScheduleDAGMILive(MachineSchedContext *C,
                        std::unique_ptr<MachineSchedStrategy> S) :
  ScheduleDAGMILive(C, std::move(S)),
  ST(MF.getSubtarget<GCNSubtarget>()),
  MFI(*MF.getInfo<SIMachineFunctionInfo>()),
  StartingOccupancy(MFI.getOccupancy()),
  MinOccupancy(StartingOccupancy), Stage(Collect), RegionIdx(0) {

  LLVM_DEBUG(dbgs() << "Starting occupancy is " << StartingOccupancy << ".\n");
  LLVM_DEBUG(dbgs() << "Nathan and Alex's starting occupancy is: " << StartingOccupancy << ".\n");
  LLVM_DEBUG(dbgs() << "Nathan and Alex's starting occupancy is ALSO ALSO: " << StartingOccupancy << ".\n");
  LLVM_DEBUG(dbgs() << "AHHHHH " << StartingOccupancy << ".\n");
}

/*
 * void GCNScheduleDAGMILive::enumerate() {
 *   targetsMet = false;
 *   bestAPRP = getAPRP(getMaxNumVGPRs())
 *   bestSchedule can just be the current layout of the scheduling region (which should just be MachineBasicBlock iterators).
 *   while(!allNodesExplored && !targetsMet) {
 * // Since we are the scheduling strategy, we will know how many SUints are left to schedule
 *      if(isLeaf()) {
 *          // update current best result
 *        if(pass == "occupancy") {
 *          if(currentAPRP <= targetAPRP)
 *              targetsMet = true;
 *          else if (currentAPRP < bestAPRP)
 *              bestAPRP = currentAPRP;
 *        } else {
 *          if(scheduleLength <= targetLength)
 *              targetsMet = true;
 *        }
 *
 * }
 */

void GCNScheduleDAGMILive::occupancyPass(SmallVector<SUnit*, 8> topRoots, SmallVector<SUnit*, 8> bottomRoots, unsigned targetPressure) {
  LLVM_DEBUG(dbgs() << "occupancy pass called\n");
  std::map<SUnit*, node*> mapSUnitToNode = setLatenciesToOne(topRoots);
  enumerate(topRoots, bottomRoots, 0, targetPressure, true);
  restoreLatencies(topRoots, mapSUnitToNode);
}

void GCNScheduleDAGMILive::ilpPass(SmallVector<SUnit*, 8> topRoots, std::vector<SUnit* > scheduleInst, int targetPressure) {
  int lowerBound = std::distance(Regions[RegionIdx].first, Regions[RegionIdx].second);
  int upperBound = satisfyLatency(scheduleInst);
  for (int targetLength = lowerBound; targetLength < upperBound - 1; targetLength ++) {
    /* run B&B, if such schedule length exist, return */
    // if (enumerate())
    //   return;
  }
}

/// schedule a instruction to RegionEnd pointer
/// move teh RegionEnd pointer to the next inst
void GCNScheduleDAGMILive::scheduleInst(MachineInstr* MI) {
  if (MI->isDebugInstr()) {
    LLVM_DEBUG(dbgs() << "Debug Instr " << *MI);
    //RegionEnd = MI->getIterator();
    //++RegionEnd;
    return;
  }

  if (MI->getIterator() != RegionEnd) {
    BB->remove(MI);
    BB->insert(RegionEnd, MI);
    if (!MI->isDebugInstr())
      LIS->handleMove(*MI, true);
  }
  // Reset read-undef flags and update them later.
  for (auto &Op : MI->operands())
    if (Op.isReg() && Op.isDef())
      Op.setIsUndef(false);
  RegisterOperands RegOpers;
  RegOpers.collect(*MI, *TRI, MRI, ShouldTrackLaneMasks, false);
  if (!MI->isDebugInstr()) {
    if (ShouldTrackLaneMasks) {
      // Adjust liveness and add missing dead+read-undef flags.
      SlotIndex SlotIdx = LIS->getInstructionIndex(*MI).getRegSlot();
      RegOpers.adjustLaneLiveness(*LIS, MRI, SlotIdx, MI);
    } else {
      // Adjust for missing dead-def flags.
      RegOpers.detectDeadDefs(*MI, *LIS);
    }
  }
  // auto RegionEndBefore = RegionEnd;
  if (RegionBegin == RegionEnd) {
    RegionBegin = MI->getIterator();
  }
  RegionEnd = MI->getIterator();
  ++RegionEnd;
  LLVM_DEBUG(dbgs() << "Scheduling " << *MI);
  // LLVM_DEBUG(dbgs() << "Region End Change: " << std::distance(RegionEndBefore, RegionEnd) << "\n");
}

unsigned GCNScheduleDAGMILive::enumerate(SmallVector<SUnit*, 8> TopRoots, SmallVector<SUnit*, 8> BottomRoots,
                                      unsigned targetLength, unsigned targetAPRP, bool isOccupanyPass)
{
  // should set regionEnd to regionStart, so we can build our own schedule (store original regionEnd)
  // keep a stack of ready queues so that we can backtrack effectively
  // keep a list of instructions (SUnit) that have been scheduled
  bool foundBetterSchedule = false;
  auto bestAPRP = getRealRegPressure().getVGPRNum(); // need to call differently to specify region?  also need to convert to APRP?
  // ReadyQueue nodesToAdd(1, "MyReadyQueue");
  std::stack<std::queue<SUnit*>> readyNodeStack; // ready nodes are nodes that have not been explored yet
  std::stack<std::deque<SUnit*>> availableNodeStack; // available nodes are nodes that are available to be scheduled given the current schedule
  std::vector<SUnit*> bestScheduledInstructions;
  std::vector<SUnit*> currentScheduledInstructions;
  readyNodeStack.emplace();
  availableNodeStack.emplace();

  for (SUnit* s: TopRoots)
  {
    // nodesToAdd.push(s);
    readyNodeStack.top().push(s);
    availableNodeStack.top().push_back(s);
  } // add possible starting SUnits
  
  MachineBasicBlock::iterator RegionBeginClone = RegionBegin;
  int schedLength = std::distance(RegionBegin, RegionEnd);

  std::vector<MachineInstr*> currentSched;
  currentSched.reserve(NumRegionInstrs);
  for (auto &I : *this) {
    currentSched.push_back(&I);
  }
  LLVM_DEBUG(dbgs() << "Before: BB begin to RegionBegin distance" << std::distance(BB->begin(), RegionBegin) <<"\n");
  LLVM_DEBUG(dbgs() << "Before: RegionBegin to BB end distance " << std::distance(RegionBegin, BB->end()) <<"\n");
  RegionEnd = RegionBegin;
  
  while (!readyNodeStack.empty()) // correct condition?
  {
    if (readyNodeStack.top().empty()) {
      LLVM_DEBUG(dbgs() << "Popping stack.\n");
      readyNodeStack.pop();
      availableNodeStack.pop();
      if (!currentScheduledInstructions.empty()){
        currentScheduledInstructions.pop_back();
        --RegionEnd;
      }
      LLVM_DEBUG(dbgs() << "New readyNodeStack size: " << readyNodeStack.size() << "\n");
      continue;
    }
    SUnit* s = readyNodeStack.top().front();
    readyNodeStack.top().pop();
    
    scheduleInst(s->getInstr());
    currentScheduledInstructions.push_back(s);

    // 1. don't call scheduleMI, but schedule instructions in the BB (see end of schedule function)
    // 2. ensure that RegionStart and RegionEnd are updated prior to checkNode call
    // TODO - fix first argument for ilp pass
    std::map<int, SUnit*> temp;
    if (checkNode(s, /*schedule*/ temp, currentScheduledInstructions, targetLength, targetAPRP, bestAPRP, isOccupanyPass))
    {
      LLVM_DEBUG(dbgs() << "current sched len: " << currentScheduledInstructions.size() << ", total len: " << static_cast<size_t>(schedLength) << "\n");
      // check leaf
      if (currentScheduledInstructions.size() == static_cast<size_t>(schedLength)) {
        unsigned regPressure = getRealRegPressure().getVGPRNum();
        if (regPressure < targetAPRP) {
          // bestScheduledInstructions = currentScheduledInstructions;
          LLVM_DEBUG(dbgs() << "Current schedule has lower RP than the target RP. Returning.\n");
          return targetAPRP;
        }
        else if (regPressure < bestAPRP) {
          bestAPRP = regPressure;
          bestScheduledInstructions = currentScheduledInstructions;
          foundBetterSchedule = true;
        } else {
          currentScheduledInstructions.pop_back();
          --RegionEnd;
          continue;
        }
      }

      availableNodeStack.emplace(availableNodeStack.top());
      for (auto it = availableNodeStack.top().begin(); it != availableNodeStack.top().end(); ++it)
      {
        if (*it == s)
        {
          availableNodeStack.top().erase(it);
          break;
        }
      }
      // availableNodeStack.top().erase(s);
      readyNodeStack.emplace(availableNodeStack.top());
      LLVM_DEBUG(dbgs() << "Adding to readyNodeStack.  New size: " << readyNodeStack.size() << "\n");
      // still promising, add successors to ready queue (not sure if this is best/right way to proceed)
      for (auto dep: s->Succs)
      {
        SUnit* succ = dep.getSUnit();
        bool hasSched = true;
        for (auto pred_dep : succ -> Preds) {
          auto it = find(currentScheduledInstructions.begin(), currentScheduledInstructions.end(), pred_dep.getSUnit());
          if (it == currentScheduledInstructions.end()) {
            hasSched = false;
          }
        }
        if (hasSched){
          readyNodeStack.top().push(succ);
          availableNodeStack.top().push_back(succ);
        }
      }
    }
    else {
      RegionEnd --;
      currentScheduledInstructions.pop_back();
    }
  }

  if (!foundBetterSchedule)
  {
    LLVM_DEBUG(dbgs() << "Didn't find better schedule - putting back the original.\n");
    //LLVM_DEBUG(dbgs() << "Original Region Begin: " << &RegionBeginClone << "\n");
    //LLVM_DEBUG(dbgs() << "Current Region Begin: " << &RegionBegin << "\n");
    LLVM_DEBUG(dbgs() << std::distance(RegionBegin, RegionBeginClone) <<"\n");
    LLVM_DEBUG(dbgs() << "After: BB Begin to RegionBegin distance " << std::distance(BB->begin(), RegionBegin) <<"\n");
    LLVM_DEBUG(dbgs() << "After: Region Begin to BB block distance " << std::distance(RegionBegin, BB->end()) <<"\n");
    RegionEnd = RegionBegin;
    for (MachineInstr *MI : currentSched) {
      scheduleInst(MI);
    }
    return bestAPRP;
  } else {
    RegionEnd = RegionBegin;
    for (SUnit *su : bestScheduledInstructions) {
      scheduleInst(su->getInstr());
    }
    return bestAPRP;
  }
}

bool GCNScheduleDAGMILive::checkNode(SUnit* node, const std::map<int, SUnit*>& scheduleSoFar, const std::vector<SUnit*>& currentScheduledInstructions, unsigned targetLength, unsigned targetAPRP, 
                                unsigned enumBestAPRP, bool isOccupancyPass) {
  if (!isOccupancyPass) {
    LLVM_DEBUG(dbgs() << "checkNode() for ilp pass.\n");
 
    /*
    * 1. tighten scheduling ranges - only needed during the ilp pass
    * Check that there is at least one cycle time within the instruction's
    * [estart, lstart] that does not have an instruction scheduled.
    */
    bool freeCycle = false;
    for (int i = estart[node]; i <= lstart[node]; ++i) {
      if (scheduleSoFar.find(i) == scheduleSoFar.end()) {
        freeCycle = true;
        break;
      }
    }
    if (!freeCycle) {
      return false;
    }

    /*
    * 2. dynamic lower bound - only needed during the ilp pass
    * Check that the schedule length so far is still less than the targetLength.
    */
    unsigned lengthSoFar = computeDLB(scheduleSoFar);
    if (lengthSoFar > targetLength) {
      return false;
    }
  }
  else {
    // LLVM_DEBUG(dbgs() << "checkNode() for occupancy pass.\n");
  }
  // 4. compute aprp
  unsigned aprp = getRealRegPressure().getVGPRNum();

  // 3. check history??? - SKIP FOR NOW
  std::vector<MachineInstr*> history(currentScheduledInstructions.size());
  std::transform(currentScheduledInstructions.begin(), currentScheduledInstructions.end(), history.begin(), [](const SUnit* s){return s->getInstr();});
  std::sort(history.begin(), history.end());
  static std::map<std::vector<MachineInstr*>, unsigned> historyMap;
  auto historyIter = historyMap.find(history);
  if(historyIter != historyMap.end()) {
    LLVM_DEBUG(dbgs() << "Find entry in history table. APRP: " << historyIter->second << "\n");
    if(historyIter->second <= aprp) {
      LLVM_DEBUG(dbgs() << "Pruned based on history table. Current APRP: " << aprp << "\n.");
      return false;
    }
  } else {
      historyMap.emplace_hint(historyIter, std::move(history), aprp);
  }

  // 5. wrap up stuff
  if (isOccupancyPass) {
    // LLVM_DEBUG(dbgs() << "checkNode() for occupancy pass just before return.\n");
    return aprp < enumBestAPRP;
  }
  else
    return aprp <= targetAPRP;
}

void GCNScheduleDAGMILive::schedule() {
    /*
     * TODO: actually implement this
     * TODO: How do we add a stall?
     * TODO: verify schedule length. SUnit::getDepth (look into mark depth dirty to trigger a re-computation)?
     * Go forth with a 281 GenPerms approach. Will likely want to make it iterative so that it better matches the paper.
     *      - Don't forget about EECS 281 genPerms() (recursive function for traversing the decision tree).
     *   TODO: How to mark things as infeasible to the parent?
     *  - Overriding the scheduling strategy probably isn't a good idea- we would have to figure out when we are generating a heuristic schedule (and use the default implementation of things like pickNode) vs when we want to pick the next node based on the BB algorithm.
     *  - Instead, don't specialize the scheduling strategy, and use the default one only during the heuristic schedule generation & ignore during BB algorithm execution.
     *  - Maybe using that table in the paper?
     * Schedule can become a state machine.
     * First, generate heuristic schedule for each region, and record metadata needed (probably APRP of that schedule or something)
     *  - Heuristic schedule is from ScheduleDAGMILive::schedule
     *  - If want to compare to AMDGPU schedule can do GCNScheduleDAGMILive::schedule (may have to inherit? IDK, sounds hard).
     *  - Need some way to get the APRP from the current schedule
     *      * Can get VGPR PRP via getRealRegPressure().getVGPRNum().
     *      * Also have ways to track occupancy via MFI
     *          - We think this implementation matches the paper: https://github.com/ROCm-Developer-Tools/llvm-project/blob/b4902bcd986ddbda30a210fc320c1fd8657e5b0d/llvm/lib/Target/AMDGPU/AMDGPUSubtarget.cpp#L643
     *          - Can change LLVM's implementation to match (right now for GFX10 it doesn't do anything).
     *          - Might use this to determine if occupancy pass improved anything.
     * Then run the Occupancy pass
     *  - apply DAG mutation (work in progress via Eric and Lisa)
     *  - run Enumerate() for given time limit
     *  - update TargetAPRP
     * Second call will run ILP pass if Occupancy pass generated different schedule
     *  - run Enumerate() for given time limit on increasing schedule lengths
     *
     */
  if (Stage == Collect) {
    // Just record regions at the first pass.
    Regions.push_back(std::make_pair(RegionBegin, RegionEnd));
    return;
  }

  std::vector<MachineInstr*> Unsched;
  Unsched.reserve(NumRegionInstrs);
  for (auto &I : *this) {
    Unsched.push_back(&I);
  }

  GCNRegPressure PressureBefore;
  if (LIS) {
    PressureBefore = Pressure[RegionIdx];

    LLVM_DEBUG(dbgs() << "Pressure before scheduling:\nRegion live-ins:";
               GCNRPTracker::printLiveRegs(dbgs(), LiveIns[RegionIdx], MRI);
               dbgs() << "Region live-in pressure:  ";
               llvm::getRegPressure(MRI, LiveIns[RegionIdx]).print(dbgs());
               dbgs() << "Region register pressure: ";
               PressureBefore.print(dbgs()));
  }

  ScheduleDAGMILive::schedule();
  Regions[RegionIdx] = std::make_pair(RegionBegin, RegionEnd);
  RescheduleRegions[RegionIdx] = false;

  buildDAGWithRegPressure();
  Topo.InitDAGTopologicalSorting();
  SmallVector<SUnit*, 8> TopRoots;
  SmallVector<SUnit*, 8> BotRoots;
  findRootsAndBiasEdges(TopRoots, BotRoots);

  if (!LIS)
    return;

  // Check the results of scheduling.
  GCNMaxOccupancySchedStrategy &S = (GCNMaxOccupancySchedStrategy&)*SchedImpl;
  auto PressureAfter = getRealRegPressure();

  LLVM_DEBUG(dbgs() << "Pressure after scheduling: ";
             PressureAfter.print(dbgs()));
  
  occupancyPass(TopRoots, BotRoots, 24);

  unsigned Occ = MFI.getOccupancy();
  unsigned WavesAfter = std::min(Occ, PressureAfter.getOccupancy(ST));
  unsigned WavesBefore = std::min(Occ, PressureBefore.getOccupancy(ST));
  LLVM_DEBUG(dbgs() << "Occupancy before scheduling: " << WavesBefore
                    << ", after " << WavesAfter << ".\n");
  if(WavesAfter > WavesBefore) {
      LLVM_DEBUG(dbgs() << "@@@ Occupancy improved!\n");
      exit(1);
  }
  if (PressureAfter.getSGPRNum() <= S.SGPRCriticalLimit &&
      PressureAfter.getVGPRNum() <= S.VGPRCriticalLimit) {
    Pressure[RegionIdx] = PressureAfter;
    LLVM_DEBUG(dbgs() << "Pressure in desired limits, done.\n");
    return;
  }

  // We could not keep current target occupancy because of the just scheduled
  // region. Record new occupancy for next scheduling cycle.
  unsigned NewOccupancy = std::max(WavesAfter, WavesBefore);
  // Allow memory bound functions to drop to 4 waves if not limited by an
  // attribute.
  if (WavesAfter < WavesBefore && WavesAfter < MinOccupancy &&
      WavesAfter >= MFI.getMinAllowedOccupancy()) {
    LLVM_DEBUG(dbgs() << "Function is memory bound, allow occupancy drop up to "
                      << MFI.getMinAllowedOccupancy() << " waves\n");
    NewOccupancy = WavesAfter;
  }
  if (NewOccupancy < MinOccupancy) {
    MinOccupancy = NewOccupancy;
    MFI.limitOccupancy(MinOccupancy);
    LLVM_DEBUG(dbgs() << "Occupancy lowered for the function to "
                      << MinOccupancy << ".\n");
  }

  unsigned MaxVGPRs = ST.getMaxNumVGPRs(MF);
  unsigned MaxSGPRs = ST.getMaxNumSGPRs(MF);
  if (PressureAfter.getVGPRNum() > MaxVGPRs ||
      PressureAfter.getSGPRNum() > MaxSGPRs)
    RescheduleRegions[RegionIdx] = true;

  if (WavesAfter >= MinOccupancy) {
    if (Stage == UnclusteredReschedule &&
        !PressureAfter.less(ST, PressureBefore)) {
      LLVM_DEBUG(dbgs() << "Unclustered reschedule did not help.\n");
    } else if (WavesAfter > MFI.getMinWavesPerEU() ||
        PressureAfter.less(ST, PressureBefore) ||
        !RescheduleRegions[RegionIdx]) {
      Pressure[RegionIdx] = PressureAfter;
      return;
    } else {
      LLVM_DEBUG(dbgs() << "New pressure will result in more spilling.\n");
    }
  }
  
  /*
  LLVM_DEBUG(dbgs() << "Attempting to revert scheduling.\n");
  RescheduleRegions[RegionIdx] = true;
  RegionEnd = RegionBegin;
  for (MachineInstr *MI : Unsched) {
    if (MI->isDebugInstr())
      continue;

    if (MI->getIterator() != RegionEnd) {
      BB->remove(MI);
      BB->insert(RegionEnd, MI);
      if (!MI->isDebugInstr())
        LIS->handleMove(*MI, true);
    }
    // Reset read-undef flags and update them later.
    for (auto &Op : MI->operands())
      if (Op.isReg() && Op.isDef())
        Op.setIsUndef(false);
    RegisterOperands RegOpers;
    RegOpers.collect(*MI, *TRI, MRI, ShouldTrackLaneMasks, false);
    if (!MI->isDebugInstr()) {
      if (ShouldTrackLaneMasks) {
        // Adjust liveness and add missing dead+read-undef flags.
        SlotIndex SlotIdx = LIS->getInstructionIndex(*MI).getRegSlot();
        RegOpers.adjustLaneLiveness(*LIS, MRI, SlotIdx, MI);
      } else {
        // Adjust for missing dead-def flags.
        RegOpers.detectDeadDefs(*MI, *LIS);
      }
    }
    RegionEnd = MI->getIterator();
    ++RegionEnd;
    LLVM_DEBUG(dbgs() << "Scheduling " << *MI);
  }
  */
  RegionBegin = Unsched.front()->getIterator();
  Regions[RegionIdx] = std::make_pair(RegionBegin, RegionEnd);

  placeDebugValues();
}

GCNRegPressure GCNScheduleDAGMILive::getRealRegPressure() const {
  GCNDownwardRPTracker RPTracker(*LIS);
  RPTracker.advance(begin(), end(), &LiveIns[RegionIdx]);
  return RPTracker.moveMaxPressure();
}

void GCNScheduleDAGMILive::computeBlockPressure(const MachineBasicBlock *MBB) {
  GCNDownwardRPTracker RPTracker(*LIS);

  // If the block has the only successor then live-ins of that successor are
  // live-outs of the current block. We can reuse calculated live set if the
  // successor will be sent to scheduling past current block.
  const MachineBasicBlock *OnlySucc = nullptr;
  if (MBB->succ_size() == 1 && !(*MBB->succ_begin())->empty()) {
    SlotIndexes *Ind = LIS->getSlotIndexes();
    if (Ind->getMBBStartIdx(MBB) < Ind->getMBBStartIdx(*MBB->succ_begin()))
      OnlySucc = *MBB->succ_begin();
  }

  // Scheduler sends regions from the end of the block upwards.
  size_t CurRegion = RegionIdx;
  for (size_t E = Regions.size(); CurRegion != E; ++CurRegion)
    if (Regions[CurRegion].first->getParent() != MBB)
      break;
  --CurRegion;

  auto I = MBB->begin();
  auto LiveInIt = MBBLiveIns.find(MBB);
  if (LiveInIt != MBBLiveIns.end()) {
    auto LiveIn = std::move(LiveInIt->second);
    RPTracker.reset(*MBB->begin(), &LiveIn);
    MBBLiveIns.erase(LiveInIt);
  } else {
    auto &Rgn = Regions[CurRegion];
    I = Rgn.first;
    auto *NonDbgMI = &*skipDebugInstructionsForward(Rgn.first, Rgn.second);
    auto LRS = BBLiveInMap.lookup(NonDbgMI);
    assert(isEqual(getLiveRegsBefore(*NonDbgMI, *LIS), LRS));
    RPTracker.reset(*I, &LRS);
  }

  for ( ; ; ) {
    I = RPTracker.getNext();

    if (Regions[CurRegion].first == I) {
      LiveIns[CurRegion] = RPTracker.getLiveRegs();
      RPTracker.clearMaxPressure();
    }

    if (Regions[CurRegion].second == I) {
      Pressure[CurRegion] = RPTracker.moveMaxPressure();
      if (CurRegion-- == RegionIdx)
        break;
    }
    RPTracker.advanceToNext();
    RPTracker.advanceBeforeNext();
  }

  if (OnlySucc) {
    if (I != MBB->end()) {
      RPTracker.advanceToNext();
      RPTracker.advance(MBB->end());
    }
    RPTracker.reset(*OnlySucc->begin(), &RPTracker.getLiveRegs());
    RPTracker.advanceBeforeNext();
    MBBLiveIns[OnlySucc] = RPTracker.moveLiveRegs();
  }
}

DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet>
GCNScheduleDAGMILive::getBBLiveInMap() const {
  assert(!Regions.empty());
  std::vector<MachineInstr *> BBStarters;
  BBStarters.reserve(Regions.size());
  auto I = Regions.rbegin(), E = Regions.rend();
  auto *BB = I->first->getParent();
  do {
    auto *MI = &*skipDebugInstructionsForward(I->first, I->second);
    BBStarters.push_back(MI);
    do {
      ++I;
    } while (I != E && I->first->getParent() == BB);
  } while (I != E);
  return getLiveRegMap(BBStarters, false /*After*/, *LIS);
}

void GCNScheduleDAGMILive::finalizeSchedule() {
    /*
     * Similar state machine as current implementation
     *
     * Not sure if need a "Collect" state or not, but can figure that out later.
     * 1) set targetAPRP to max occupancy /"initial target occupancy"
     * 2) For each region, call schedule() on the region (will want to basically copy/pasta the existing do while loop)
     *  - This should run Occupancy pass and update TargetAPRP
     * 3) For each region call schedule() (should be done via copy/pasta-ed loop)
     *  - This should run ILP pass
     */
  LLVM_DEBUG(dbgs() << "Calling our finalizeSchedule.\n");
  LLVM_DEBUG(dbgs() << "All regions recorded, starting actual scheduling.\n");
  
  GCNMaxOccupancySchedStrategy &S = (GCNMaxOccupancySchedStrategy&)*SchedImpl;
  
  LiveIns.resize(Regions.size());
  Pressure.resize(Regions.size());
  RescheduleRegions.resize(Regions.size());
  RescheduleRegions.set();

  if (!Regions.empty())
    BBLiveInMap = getBBLiveInMap();

  std::vector<std::unique_ptr<ScheduleDAGMutation>> SavedMutations;

  do {
    Stage++;
    RegionIdx = 0;
    MachineBasicBlock *MBB = nullptr;

    if (Stage > InitialSchedule) {
      if (!LIS)
        break;

      // Retry function scheduling if we found resulting occupancy and it is
      // lower than used for first pass scheduling. This will give more freedom
      // to schedule low register pressure blocks.
      // Code is partially copied from MachineSchedulerBase::scheduleRegions().

      if (Stage == UnclusteredReschedule) {
        if (RescheduleRegions.none())
          continue;
        LLVM_DEBUG(dbgs() <<
          "Retrying function scheduling without clustering.\n");
      }

      if (Stage == ClusteredLowOccupancyReschedule) {
        if (StartingOccupancy <= MinOccupancy)
          break;

        LLVM_DEBUG(
            dbgs()
            << "Retrying function scheduling with lowest recorded occupancy "
            << MinOccupancy << ".\n");

        S.setTargetOccupancy(MinOccupancy);
      }
    }

    if (Stage == UnclusteredReschedule)
      SavedMutations.swap(Mutations);

    for (auto Region : Regions) {
      if (Stage == UnclusteredReschedule && !RescheduleRegions[RegionIdx]) {
        ++RegionIdx;
        continue;
      }

      RegionBegin = Region.first;
      RegionEnd = Region.second;

      if (RegionBegin->getParent() != MBB) {
        if (MBB) finishBlock();
        MBB = RegionBegin->getParent();
        startBlock(MBB);
        if (Stage == InitialSchedule)
          computeBlockPressure(MBB);
      }

      unsigned NumRegionInstrs = std::distance(begin(), end());
      enterRegion(MBB, begin(), end(), NumRegionInstrs);

      // Skip empty scheduling regions (0 or 1 schedulable instructions).
      if (begin() == end() || begin() == std::prev(end())) {
        exitRegion();
        continue;
      }

      LLVM_DEBUG(dbgs() << "********** MI Scheduling **********\n");
      LLVM_DEBUG(dbgs() << MF.getName() << ":" << printMBBReference(*MBB) << " "
                        << MBB->getName() << "\n  From: " << *begin()
                        << "    To: ";
                 if (RegionEnd != MBB->end()) dbgs() << *RegionEnd;
                 else dbgs() << "End";
                 dbgs() << " RegionInstrs: " << NumRegionInstrs << '\n');

      schedule();

      exitRegion();
      ++RegionIdx;
    }
    finishBlock();

    if (Stage == UnclusteredReschedule)
      SavedMutations.swap(Mutations);
  } while (Stage != LastStage);
  
}

std::map<SUnit*, node*> GCNScheduleDAGMILive::setLatenciesToOne(SmallVector<llvm::SUnit *, 8> &topRoots) {
    // a mapping from original graph to the clone graph
    std::map<SUnit*, node*> mapSUnitToNode;

    std::queue<SUnit*> queue;
    std::vector<SUnit*> visited;

    for (auto it =  topRoots.begin(); it != topRoots.end(); ++it) {
        queue.push(*it);
        visited.push_back(*it);
    }

    // initialize a one-to-one mapping from the original SUnit to node
    while (!queue.empty()) {
        SUnit* sunit = queue.front();
        queue.pop();

        struct node* clone = new node;
        mapSUnitToNode.insert({sunit, clone});
        SmallVector<SDep, 4> succs = sunit -> Succs;
        for (auto it = succs.begin(); it != succs.end(); ++it) {
            SUnit* succ = it->getSUnit();
            if (find(visited.begin(), visited.end(), succ) == visited.end()){
                queue.push(succ);
                visited.push_back(succ);
            }
        }
    }

    LLVM_DEBUG(dbgs() << "mapSUnitToNode size: " << mapSUnitToNode.size() << "\n");

    visited.clear();
    for (auto it =  topRoots.begin(); it != topRoots.end(); ++it) {
        queue.push(*it);
        visited.push_back(*it);
    }

    // build the clone graph and set all original latencies to one
    while (!queue.empty()) {
        SUnit* sunit = queue.front();
        queue.pop();

        node* currentNode = mapSUnitToNode[sunit];
        SmallVector<SDep, 4> succs = sunit -> Succs;
        for (auto it = succs.begin(); it != succs.end(); ++it) {
            SUnit* succ = it->getSUnit();
            int latency = it->getLatency();
            node* succNode = mapSUnitToNode[succ];

            // set current node as successor's pred
            LLVM_DEBUG(dbgs() << static_cast<void*>(succNode) << "\n");
            // LLVM_DEBUG(dbgs() << typeid(succNode->preds).name() << " " << typeid(succNode->predsLatencies).name() << " " << latency << "\n");
            succNode->preds.push_back(currentNode);
            succNode->predsLatencies.push_back(latency);
            
            // set successor as current node's succ
            currentNode->succs.push_back(succNode);
            currentNode->succsLatencies.push_back(latency);

            // set latency to 1
            it->setLatency(1);

            // push succ if it is not in the queue or visited
            if (find(visited.begin(), visited.end(), succ) == visited.end()){
                queue.push(succ);
                visited.push_back(succ);
            }
        }
    }
    return mapSUnitToNode;
}

void GCNScheduleDAGMILive::restoreLatencies(SmallVector<llvm::SUnit *, 8> &topRoots, std::map<SUnit*, node*> mapSUnitToNode) {
  std::queue<SUnit*> queue;
  std::vector<SUnit*> visited;

  for (auto it : topRoots) {
      queue.push(it);
      visited.push_back(it);
  }

  while (!queue.empty()) {
      SUnit* sunit = queue.front();
      queue.pop();

      SmallVector<SDep, 4> succs = sunit -> Succs;
      node* currentNode = mapSUnitToNode[sunit];
      for (auto dep : succs) {
          SUnit* succ = dep.getSUnit();
          node* succNode = mapSUnitToNode[succ];
          int latency = currentNode->getSuccLatency(succNode);
          dep.setLatency(latency);

          if (find(visited.begin(), visited.end(), succ) == visited.end()){
              queue.push(succ);
              visited.push_back(succ);
          }

          
      }
  }

  for (auto it = mapSUnitToNode.begin(); it != mapSUnitToNode.end(); ++it)
  {
    delete it->second;
  }

}

/// Compute how many stall need to be inserted to satisfy the latency
int GCNScheduleDAGMILive::satisfyLatency(std::vector<SUnit*> schedule) {
    int current_cycle = 0;
    unsigned index = 0;

    std::vector<SDep> sched;
    std::vector<int> cycle_count;
    // std::unordered_map<SDep, int> sched_sdep;

    while (index < schedule.size()) {
        SUnit* sunit = schedule[index];
        SmallVector<SDep, 4> preds = sunit -> Preds;

        bool can_schedule = true;
        for (auto pred : preds) {
            auto it = find(sched.begin(), sched.end(), pred);
            if (it != sched.end()) {
                if (cycle_count[it-sched.begin()] > 0){
                    can_schedule = false;
                }

            }
        }
        if (can_schedule) {
            for (auto succ : sunit -> Succs) {
                int latency = succ.getLatency();
                sched.push_back(succ);
                cycle_count.push_back(latency);
            }
            // schedule next inst
            index ++;
        }

        // increase cycle
        current_cycle ++;

        // decrease all scheduled latency
        for (unsigned i = 0; i < cycle_count.size(); i ++) {
            cycle_count[i] --;
        }
    }
    return current_cycle;
}

void GCNScheduleDAGMILive::computeEstart(SmallVector<SUnit*, 8> topRoots) {
  std::deque<SUnit*> nodes(topRoots.begin(), topRoots.end());

  while (!nodes.empty()) {
    SUnit* node = nodes.front();
    nodes.pop_front();
    if (node->NumPreds == 0) {
      estart[node] = 0;
    }
    else {
      SmallVector<SDep, 4> preds = node->Preds;
      unsigned maxVal = 0;
      bool allPreds = true;
      for (auto it = preds.begin(); it != preds.end(); ++it) {
        if (estart.find(it->getSUnit()) == estart.end()) {
          allPreds = false;
          break;
        }
        maxVal = (maxVal > estart[it->getSUnit()] + it->getLatency()) ? maxVal : (estart[it->getSUnit()] + it->getLatency());
      }
      if (allPreds) {
        estart[node] = maxVal;
        SmallVector<SDep, 4> succs = node->Succs;
        for (auto it =  succs.begin(); it != succs.end(); ++it) {
          if (find(nodes.begin(), nodes.end(), it->getSUnit()) == nodes.end()) 
            nodes.push_back(it->getSUnit());
        }
      }
      else {
        nodes.push_back(node);
      }
    }
  }
}

void GCNScheduleDAGMILive::computeLstart(SmallVector<SUnit*, 8> bottomRoots, int maxEstart) {
  std::deque<SUnit*> nodes(bottomRoots.begin(), bottomRoots.end());

  while (!nodes.empty()) {
    SUnit* node = nodes.front();
    nodes.pop_front();
    if (node->NumSuccs == 0) {
      lstart[node] = maxEstart;
    }
    else {
      SmallVector<SDep, 4> succs = node->Succs;
      unsigned minVal = 0;
      bool allSuccs = true;
      for (auto it = succs.begin(); it != succs.end(); ++it) {
        if (lstart.find(it->getSUnit()) == lstart.end()) {
          allSuccs = false;
          break;
        }
        minVal = (minVal < lstart[it->getSUnit()] - it->getLatency()) ? minVal : (lstart[it->getSUnit()] - it->getLatency()); 
      }
      if (allSuccs) {
        lstart[node] = minVal;
        SmallVector<SDep, 4> preds = node->Preds;
        for (auto it =  preds.begin(); it != preds.end(); ++it) {
          if (find(nodes.begin(), nodes.end(), it->getSUnit()) == nodes.end()) 
            nodes.push_back(it->getSUnit());
        }
      }
      else {
        nodes.push_back(node);
      }
    }
  }
}


class cmp {
  public:
    bool operator()(const std::pair<SUnit*, int> &a, const std::pair<SUnit*, int> &b) const { 
      return a.second < b.second; 
    } 
};

unsigned GCNScheduleDAGMILive::computeDLB(std::map<int, SUnit*> scheduleSoFar) {
  std::map<SUnit*, int> tempLstart;
  for (auto it = scheduleSoFar.begin(); it != scheduleSoFar.end(); ++it) {
    tempLstart[it->second] = lstart[it->second];
  }

  // 4. sort operations in order of increasing ALAP
  std::set<std::pair<SUnit*, int>, cmp> ops(tempLstart.begin(), tempLstart.end()); 

  // 5. schedule each operation
  std::map<int, SUnit*> schedule;
  int maxDelay = 0;
  for (auto it = ops.begin(); it != ops.end(); ++it) {
    int opTime = estart[it->first];

    while (true) {
      if (schedule.find(opTime) == schedule.end()) {
        schedule[opTime] = it->first;
        break;
      }
      ++opTime;
    }

    maxDelay = (maxDelay > opTime - lstart[it->first]) ? maxDelay : opTime - lstart[it->first]; 
  }

  // 6. dynamic lower bound = maxDelay + criticalPathDelay
  int criticalPathDelay = tempLstart.rbegin()->second;

  return maxDelay + criticalPathDelay;
}
