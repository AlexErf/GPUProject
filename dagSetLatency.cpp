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

    std::deque<SUnit*> queue;
    std::vector<SUnit*> visited;

    for (auto it =  topRoots.begin(); it != topRoots.end(); ++it) {
        queue.push(*it);
        visited.push_back(*it);
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
            if (find(visited.begin(), visited.end(), succ) == visited.begin()){
                queue.push(succ);
                visited.push_back(succ)
            }
        }
    }

    visited.clear();
    for (auto it =  topRoots.begin(); it != topRoots.end(); ++it) {
        queue.push(*it);
        visited.push_back(*it);
    }

    // build the clone graph and set all original latencies to one
    while (!queue.empty()) {
        SUnit* sunit = queue.front();
        queue.pop();

        node& currentNode = mapSUnitToNode[sunit];
        SmallVector<SDep, 4> succs = sunit -> Succs;
        for (auto it = succs.begin(); it != succs.end(); ++it) {
            SUnit* succ = it->getSUnit();
            int latency = it->getLatency();
            node& succNode = mapSUnitToNode[succ];

            // set current node as successor's pred
            succNode.preds.push_back(currentNode);
            succNode.predsLatencies.push_back(latency);

            // set successor as current node's succ
            currentNode.succs.push_back(succNode);
            currentNode.succsLatencies.push_back(latency);

            // set latency to 1
            it->setLatency(1);

            // push succ if it is not in the queue or visited
            if (find(visited.begin(), visited.end(), succ) == visited.end()){
                queue.push(succ);
                visited.push_back(succ)
            }
        }
    }
    return mapSUnitToNode;
}

static void restoreLatencies(std::vector<SUnit*> topRoots, std::map<SUnit*, node> mapSUnitToNode) {
    std::queue<SUnit*> queue;
    std::vector<SUnit*> visited;

    for (auto it : topRoots) {
        queue.push(*it);
        visited.push(*it);
    }

    while (!queue.empty()) {
        SUnit* sunit = queue.front();
        queue.pop();

        SmallVector<SDep, 4> succs = it -> Succs;
        node currentNode = mapSUnitToNode[it];
        for (auto dep : succs) {
           SUnit* succ = dep.getSUnit();
           node succNode = mapSUnitToNode[succ];
           int latency = currentNode.getSuccLatency(succNode);
           dep.setLatency(latency);

           if (find(visited.begin(), visited.end(), succ) == visited.end()){
                queue.push(succ);
                visited.push_back(succ)
            }
        }
    }
}