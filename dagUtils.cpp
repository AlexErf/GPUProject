static void computeEstart(map<SUnit*, int> &estart, SUnit* node) {
  if (node->NumPreds == 0) {
    estart[node] = 0;
  }
  else {
    SmallVector<SDep, 4> preds = node->Preds;
    int maxVal = 0;
    for (auto it = preds.begin(); it != preds.end(); ++it) 
      maxVal = max(maxVal, estart[it->getSUnit()] + it->getLatency());
    estart[node] = maxVal;
  }
}

static void computeLstart(map<SUnit*, int> &lstart, SUnit* node, int maxEstart) {
  if (node->NumSuccs == 0) {
    lstart[node] = maxEstart;
  }
  else {
    SmallVector<SDep, 4> succs = node->Succs;
    int minVal = 0;
    for (auto it = succs.begin(); it != succs.end(); ++it) 
      minVal = min(minVal, lstart[it->getSUnit()] - it->getLatency());
    lstart[node] = minVal;
  }
}

bool cmp(pair<SUnit*, int> &a, pair<SUnit*, int> &b) { 
  return a.second < b.second; 
} 

static int computeDLB(SmallVector<SUnit*, 8> topRoots, SmallVector<SUnit*, 8> botRoots) {
  map<SUnit*, int> estart;
  map<SUnit*, int> lstart;

  // 3. compute ASAP/estart and ALAP/lstart for each SUnit/node
  for (auto it = topRoots.begin(); it != topRoots.end(); ++it) 
    computeEstart(estart, *it);

  int maxEstart = 0;
  for (auto it = estart.begin(); it != estart.end(); ++it) 
    maxEstart = max(maxEstart, it->second);

  for (auto it = botRoots.begin(); it != botRoots.end(); ++it) 
    computeLstart(lstart, *it, maxEstart);

  // 4. sort operations in order of increasing ALAP
  set<pair<SUnit*, int>, cmp> ops(lstart.begin(), lstart.end()); 

  // 5. schedule each operation
  map<int, SUnit*> schedule;
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

    maxDelay = max(maxDelay, opTime - lstart[it->first]);
  }

  // 6. dynamic lower bound = maxDelay + criticalPathDelay
  int criticalPathDelay = schedule.rbegin()->first;

  return maxDelay + criticalPathDelay;
}