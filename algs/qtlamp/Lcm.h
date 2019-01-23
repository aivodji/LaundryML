#ifndef _LCM_H_
#define _LCM_H_

#include <iostream>
#include <unordered_set>
#include <map>
#include <algorithm>
#include <cassert>
#include <utility>
#include "Database.h"
#include "OccurenceDeriver.h"
#include "tree.h"

using namespace std;

class Lcm {
  ostream &out;
public:
  int maxItem;     // maximum item
  int minItem;     // minimum item
  int max_pat;     // maximum size of itemsets

  vector<int> totalItem;
  Lcm(ostream &out, int _max_pat);
  ~Lcm();
  pair<vector<vector<int> >, Trie> RunLcm(Database database, int min_sup, bool build_trie);
  vector<vector<int> > LcmIter(Database &database, int min_sup, vector<int> &itemsets, vector<int> &transactionList, vector<int> &labelledTransactionList, OccurenceDeriver &occ, vector<int> &freqList, bool build_trie, Trie &trie);
  void PrintItemsets(const vector<int> &itemsets, const OccurenceDeriver &occ);
  int CalcurateCoreI(const vector<int> &itemsets, const vector<int> &freqList);
  bool PpcTest(const Database &database, vector<int> &itemsets, vector<int> &transactionList, int item, vector<int> &newTransactionList);
  void MakeClosure(const Database &database, vector<int> &transactionList, vector<int> &q_sets, vector<int> &itemsets, int item);
  bool CheckItemInclusion(const Database &database, vector<int> &transactionList, int item);
  vector<int> CalcTransactionList(const Database &database, const vector<int> &transactionList, int item);
  void CalcTransactionList(const Database &database, const vector<int> &transactionList, int item, vector<int> &newTransactionList);
  void UpdateTransactionList(const Database &database, const vector<int> &transactionList, const vector<int> &labelledTransactionList, const vector<int> &q_sets, int item, vector<int> &newTransactionList, vector<int> &newLabelledTransactionList);
  void UpdateFreqList(const Database &database, const vector<int> &transactionList, const vector<int> &gsub, vector<int> &freqList, int freq, vector<int> &newFreq);
  void UpdateOccurenceDeriver(const Database &database, const vector<int> &transactionList, OccurenceDeriver &occurence);
  map<vector<int>, pair<int, int> > getFrequencies(Database database, const set<vector<int> > &patterns, const Trie &trie);
  void getFrequenticesIter(Database &database, const set<vector<int> > &patterns,
	  vector<int> &itemsets, vector<int> &transactionList, OccurenceDeriver &occ, vector<int> &freqList,
	  pair<int,int> curFreq, map<vector<int>, pair<int, int> > &freqs, const Trie &trie);
  double GetPvalueLevelExcludeParents(const vector<int> &occ_i, const vector<int> &occ_labelled_i,
    double q, double a, bool mode, double passRate, const Database &database);
};
#endif // _LCM_H_
