static map<SUnit*, int> computeEstart(deque<SUnit*> nodes) {
  map<SUnit*, int> estart;

  while (!nodes.empty()) {
    SUnit* node = nodes.pop_front();
    if (node->NumPreds == 0) {
      estart[node] = 0;
    }
    else {
      SmallVector<SDep, 4> preds = node->Preds;
      int maxVal = 0;
      bool allPreds = true;
      for (auto it = preds.begin(); it != preds.end(); ++it) {
        if (estart.find(it->getSUnit()) == estart.end()) {
          allPreds = false;
          break;
        }
        maxVal = max(maxVal, estart[it->getSUnit()] + it->getLatency());
      }
      if (allPreds) {
        estart[node] = maxVal;
        SmallVector<SDep, 4> succs = node->Succs;
        for (auto it =  succs.begin(); it != succs.end(); ++it) {
          if (find(nodes.begin(), nodes.end(), *it) == nodes.end()) 
            nodes.push_back(*it);
        }
      }
      else {
        nodes.push_back(node);
      }
    }
  }
  
  return estart;
}

static map<SUnit*, int> computeLstart(deque<SUnit*> nodes, int maxEstart) {
  map<SUnit*, int> lstart;

   while (!nodes.empty()) {
    SUnit* node = nodes.pop_front();
    if (node->NumSuccs == 0) {
      lstart[node] = maxEstart;
    }
    else {
      SmallVector<SDep, 4> succs = node->Succs;
      int minVal = 0;
      bool allSuccs = true;
      for (auto it = succs.begin(); it != succs.end(); ++it) {
        if (lstart.find(it->getSUnit()) == lstart.end()) {
          allSuccs = false;
          break;
        }
        minVal = min(minVal, lstart[it->getSUnit()] - it->getLatency());
      }
      if (allSuccs) {
        lstart[node] = minVal;
        SmallVector<SDep, 4> preds = node->Preds;
        for (auto it =  preds.begin(); it != preds.end(); ++it) {
          if (find(nodes.begin(), nodes.end(), *it) == nodes.end()) 
            nodes.push_back(*it);
        }
      }
      else {
        nodes.push_back(node);
      }
    }
  }

  return lstart;
}

bool cmp(pair<SUnit*, int> &a, pair<SUnit*, int> &b) { 
  return a.second < b.second; 
} 

static int computeDLB(SmallVector<SUnit*, 8> topRoots, SmallVector<SUnit*, 8> botRoots) {
  map<SUnit*, int> estart;
  map<SUnit*, int> lstart;

  // 3. compute ASAP/estart and ALAP/lstart for each SUnit/node
  queue<SUnit*> nodes(topRoots.begin(), topRoots.end());
  max<SUnit*, int> estart = computeEstart(nodes);

  int maxEstart = 0;
  for (auto it = estart.begin(); it != estart.end(); ++it) 
    maxEstart = max(maxEstart, it->second);

  max<SUnit*, int> lstart = computeLstart(nodes, maxEstart);

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
