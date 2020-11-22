#include <iostream>
#include <vector>
#include <queue>
#include <map> 
#include "llvm/CodeGen/ScheduleDAG.h"

using namespace llvm;

struct node {
    std::vector<node> preds;
    std::vector<int> predsLatencies;

    std::vector<node> succs;
    std::vector<int> succsLatencies;

    /// a helper function to get the succ latency
    int getSuccLatency(node succ) {
        auto it = find(succs.begin(), succs.end(), succs);

        if (it != succs.end()) {
            int index = it - succs.begin();
            return succsLatencies[index];
        }
    }

    /// a helper function to get the pred latency
    int getPredLatency(node pred) {
        auto it = find(preds.begin(), preds.end(), pred);

        if (it != preds.end()) {
            int index = it - preds.begin();
            return predsLatencies[index];
        }
    }
};

static std::map<SUnit*, node> setLatenciesToOne(std::vector<SUnit*> topRoots) {
    // a mapping from original graph to the clone graph
    std::map<SUnit*, node> mapSUnitToNode;

    std::queue<SUnit*> queue;

    for (auto it =  topRoots.begin(); it != topRoots.end(); ++it) {
        queue.push(*it);
    }

    // initialize a one-to-one mapping from the original SUnit to node
    while (!queue.empty()) {
        SUnit* sunit = queue.front();
        queue.pop();

        struct node clone;
        mapSUnitToNode.insert({sunit, clone});
        SmallVector<SDep, 4> succs = sunit -> Succs;
        for (auto it = succs.begin(); it != succs.end(); ++it) {
            SUnit* succ = it->getSUnit();
            queue.push(succ);
        }
    }

    for (auto it =  topRoots.begin(); it != topRoots.end(); ++it) {
        queue.push(*it);
    }

    // build the clone graph and set all original latencies to one
    while (!queue.empty()) {
        SUnit* sunit = queue.front();
        queue.pop();

        node currentNode = mapSUnitToNode[sunit];
        SmallVector<SDep, 4> succs = sunit -> Succs;
        for (auto it = succs.begin(); it != succs.end(); ++it) {
            SUnit* succ = it->getSUnit();
            int latency = it->getLatency();
            node succNode = mapSUnitToNode[succ];

            // set current node as successor's pred
            succNode.preds.push_back(currentNode);
            succNode.predsLatencies.push_back(latency);

            // set successor as current node's succ
            currentNode.succs.push_back(succNode);
            currentNode.succsLatencies.push_back(latency);

            // set latency to 1
            it->setLatency(1);

            queue.push(succ);
        }
    }
    return mapSUnitToNode;
}