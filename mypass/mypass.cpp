#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static void schedule(/*kernal*/);
static void occupancyPass(/*kernal, targetAPRP*/);
static void ilpPass(/*region, targetAPRP*/);
static int satisfyLatencies(/*region.DDG, region.bestSched*/);
static int computeLowerBound(/*region.DDG*/);
static bool enumerate(/*region, targetLength, targetAPRP, pass*/);
static void checkNode(/*node, targetLength, targetAPRP, enumBestAPRP, pass*/);

namespace {
struct Hello : public FunctionPass {
  static char ID;
  Hello() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    // errs() << "Hello: ";
    errs() << "run 2-pass B&B on: ";
    errs().write_escaped(F.getName()) << '\n';
    schedule();
    return false;
  }
}; // end of struct Hello
}  // end of anonymous namespace

// main function for schedule
static void schedule(/*kernal*/) {
  // /* Initial the target occupancy. Normally the GPU's maximum occupancy */
  // targetAPRP = getTargetAPRP();

  // /* for each region in kernal. Run the occupancy pass and update targetAPRP */
  // for (auto* region : kernal) {
  //   occupancyPass(region, targetAPRP);   /* run occupancy pass */
  //   targetAPRP = region.bestSched.APRP;  /* update the targetAPRP to the current best APRP (this will be the bottleneck) */
  // }

  // /* for each region in kernal. Run the ILP pass with the targetAPRP */
  // for (auto* region : kernal) {
  //   ilpPass(region, targetAPRP);
  // }
}

static void occupancyPass(/*region, targetAPRP*/) {
  // /* find the heuristic schdule as the best schedule (such as LLVM’s generic heuristic) */
  // region.bestSched = findHeuristicSched()

  // /* 
  //   if the heuristic schdule has a higher APRP then the targetAPRP,
  //   we need to run the B&B algorithm and try to find a better one.
  // */
  // if (region.bestSched.APRP > targetAPRP) {
  //  /* 
  //    set all latencies to 1.
  //    ignore the latencies between instructions.
  //    e.g. schedule length = number of inst
  //  */
  //   setAllLatenciesToOne(region.DDG)
  //   enumerate(region, region.instCnt, targetAPRP, "occupancy")
  // }
}

static void ilpPass(/*region, targetAPRP*/) {
  // /* the upper bound of the schedule length */
  // upperBoundScheLength = satisfyLatencies(region.DDG, region.bestSched)

  // /* the lower bound of the schedule length */
  // lowerBoundScheLength = computeLowerBound(region.DDG)

  // /* Simply return if the best schedule is the lower bound */
  // if (region.bestSched.length == upperBoundScheLength)
  //  return;

  // /* otherwise, run B&B for each schedule length between the lower bound and upper bound */
  // for (int targetLength = lowerBoundScheLength; targetLength < upperBoundScheLength - 1; targetLength ++) {
  //   /* run B&B, if such schedule length exist, return */
  //   if (enumerate(region, targetLength, targetAPRP, "ilp"))
  //     return;
  // }
}

static int satisfyLatencies(/*region.DDG, region.bestSched*/) {
  /* 
    The upper bound is computed by adding stalls in the best schedule computed in the occupancy pass.
    e.g. the schedule with the best APRP.
    This schedule will more likely be invalid since we didn't consider the lantencies.
    By adding stalls, we can get a trivial feasible schedule that satisfy the best APRP.
    This trivial schedule will be our lower bound.
  */
  return 0;
}

static int computeLowerBound(/*region.DDG*/) {
  /* 
    We use the algorithm of Langevin and Cerny to compute the lower bound.
    See https://dl.acm.org/doi/10.1145/238997.239002.
    I think if we can't figure it out, we can use the number of instruction as the lower bound.
  */
  return 0;
}

static bool enumerate(/*region, targetLength, targetAPRP, pass*/) {
  bool targetsMet = false;

  // /* set the default best sched as the current best sched */
  // enumBestAPRP = region.bestSched.APRP
  // enumBestSched = region.bestSched

  // /* set current node as root node (which is an empty node) */
  // currentNode = rootNode

  // /* enumerate */
  // while (!allNodesExplored && !targetsMet) {
    
  //   /* if node is leaf (e.g. end of schedule) */
  //   if (isLeaf(currentNode)) {
  //     /* update current best result */
  //     if (pass == "occupancy") {
  //       if (enumAPRP <= targetAPRP)
  //         targetsMet = true;
  //       else if (enumAPRP < enumBestAPRP)
  //         enumBestAPRP = enumAPRP
  //     } else if (pass == "ilp") {
  //       if (enumLength <= targetLength)
  //         targetsMet = true;
  //     }

  //     /* back track */
  //     currentNode.parent.setNotFeasible(currentNode);
  //     currentNode = currentNode.parent
  //   } 

  //   /* if node is root and no feasible node */
  //   else if (currentNode == rootNode && currentNode.hasNoNextFeasibleNode()) {
  //     allNodesExplored = true;
  //   }

  //   /* else if no feasible child left. go backward. */
  //   else if (currentNode.hasNoNextFeasibleNode()) {
  //     /* back track */
  //     currentNode.parent.setNotFeasible(currentNode);
  //     currentNode = currentNode.parent
  //   }

  //   /* otherwise, check next feasible child */
  //   else {
  //     node = currentNode.nextFeasibleNode()
  //     /* Check if feasible. If so, move forward */
  //     if (checkNode(node, targetLength, targetAPRP, enumBestAPRP, pass)) {
  //       /* move forward */
  //       currentNode = node;
  //     } else {
  //       /* mark this node as not feasible */
  //       currentNode.setNotFeasible(node);
  //     }
  //   }
  // }

  // if (targetsMet)
  //   region.bestSched = enumBestSched

  return targetsMet;
}

static void checkNode(/*node, targetLength, targetAPRP, enumBestAPRP, pass*/) {

  // /* check if the current inst can be filled in this cycle */
  // feasible = tightenSchedRanges(node)
  // if (!feasible)
  //   return false;
  
  // /* make sure the dynamic lower bound is less then the targetLength */
  // dynamicLowerBound = computeDLB(node)
  // if (dynamicLowerBound > targetLength)
  //   return false;
  
  // /* use the history information to prove that the current node could lead to a better schedule */
  // if (checkHistory(node) == false)
  //   return false;
  
  // APRP = computeAPRP(node)

  // if (pass == "occupancy")
  //   /* for occupancy pass, the current APRP has to be less than the current best APRP. */
  //   return APRP < enumBestAPRP;
  // else 
  //   /* for ILP pass, the current APRP cannot be larger than the target APRP. */
  //   return APRP <= targetAPRP;
}

char Hello::ID = 0;
static RegisterPass<Hello> X("hello", "Hello World Pass",
                             false /* Only looks at CFG */,
                             false /* Analysis Pass */);