#include <map> 
#include "llvm/CodeGen/ScheduleDAG.h"

using namespace llvm;

/// Compute how many stall need to be inserted to satisfy the latency
static int satisfyLatency(std::vector<SUnit*> schedule) {
    int current_cycle = 0;
    int index = 0;

    std::map<SDep, int> sched_sdep;

    while (index < schedule.size()) {
        SUnit* sunit = schedule[index];
        SmallVector<SDep, 4> preds = sunit -> Preds;

        bool can_schedule = true;
        for (auto pred : preds) {
            if (sched_sdep[pred] > 0) {
                can_schedule = false;
            }
        }
        if (can_schedule) {
            for (auto succ : sunit -> Succs) {
                int latency = succ.getLatency();
                sched_sdep.insert(std::make_pair(succ, latency));
            }
            // schedule next inst
            index ++;
        }

        // increase cycle
        current_cycle ++;

        // decrease all scheduled latency
        for (std::pair<SDep, int> element : sched_sdep) {
            sched_sdep[element.first] --;
        }
    }
    return current_cycle;
}