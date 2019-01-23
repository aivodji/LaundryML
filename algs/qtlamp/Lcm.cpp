#include "Lcm.h"

/************************************************************************
 * Constructor
 ************************************************************************/
Lcm::Lcm(ostream &_out, int _max_pat) 
: out(_out), max_pat(_max_pat) {}

/************************************************************************
 * Destructor
 ************************************************************************/
Lcm::~Lcm() {}

/*************************************************************************
 * run lcm
 *************************************************************************/
pair<vector<vector<int> >, Trie> Lcm::RunLcm(Database database, int min_sup, bool build_trie) {
  OccurenceDeriver occ(database);
  maxItem = database.GetMaxItem();
  minItem = database.GetMinItem();
  Trie trie;

  vector<int> itemsets;
  vector<int> transactionList;
  vector<int> labelledTransactionList;
  for (int i = 0; i < database.GetNumberOfTransaction(); i++) {
    transactionList.push_back(i);
    if(database.database[i].label){
      labelledTransactionList.push_back(i);
    }
  }
  
  totalItem = database.GetItemset();
  vector<int> freqList;
  cout << "lcmiter start" << endl;
  vector<vector<int> > patterns = LcmIter(database, min_sup, itemsets, transactionList, labelledTransactionList, occ, freqList, build_trie, trie);
  return make_pair(patterns, trie);
}

/*************************************************************************
 * main function
 *************************************************************************/
vector<vector<int> >  Lcm::LcmIter(Database &database, int min_sup, vector<int> &itemsets,
  vector<int> &transactionList, vector<int> &labelledTransactionList, OccurenceDeriver &occ,
  vector<int> &freqList, bool build_trie, Trie &trie) {
  
  vector<vector<int> >  importantPatterns;
  importantPatterns.push_back(itemsets);
  if(itemsets.size() && build_trie){
    trie.insert(itemsets);
  }

  int core_i = CalcurateCoreI(itemsets, freqList);
  // database reduction
  // database.Reduction(transactionList, core_i); 
  // database.Print(out);
  
  /*
    Compute the frequency of each pattern P \cup {i}, i > core_i(P) 
    by Occurence deliver with P and Occ;
  */
  vector<int>::iterator iter = lower_bound(totalItem.begin(), totalItem.end(), core_i);
  vector<int> freq_i;
  for (int i = *iter; iter != totalItem.end(); iter++, i = *iter) {
      if ((int)occ.table[i].size() >= min_sup && binary_search(itemsets.begin(), itemsets.end(), i) == false) 
	freq_i.push_back(i);
  }

  vector<int> newTransactionList;
  vector<int> newLabelledTransactionList;
  vector<int> q_sets;
  vector<int> newFreqList;
  for (vector<int>::iterator freq = freq_i.begin(); freq != freq_i.end(); freq++) {
    newTransactionList.clear();
    newLabelledTransactionList.clear();
    if (PpcTest(database, itemsets, transactionList, *freq, newTransactionList)) {
      q_sets.clear();
      MakeClosure(database, newTransactionList, q_sets, itemsets, *freq);
      if (max_pat == 0 || (int)q_sets.size() <= max_pat) {
        newTransactionList.clear();
        newLabelledTransactionList.clear();
        UpdateTransactionList(database, transactionList, newLabelledTransactionList, q_sets, *freq, newTransactionList, newLabelledTransactionList);
        UpdateOccurenceDeriver(database, newTransactionList, occ);
        newFreqList.clear();
//        cout << "hoge end" << endl;
        UpdateFreqList(database, transactionList, q_sets, freqList, *freq, newFreqList);
        vector<vector<int> > temp = LcmIter(database, min_sup, q_sets, newTransactionList, newLabelledTransactionList, occ, newFreqList, build_trie, trie);
        for(vector<vector<int> >::iterator it=temp.begin();it!=temp.end();++it){
          importantPatterns.push_back(*it);
        }
      }
    }
  }

  return importantPatterns;
}

/*************************************************************************
 * Update Freq List
 *************************************************************************/
inline void Lcm::UpdateFreqList(const Database &database, const vector<int> &transactionList, const vector<int> &gsub, vector<int> &freqList, int freq, vector<int> &newFreq) {
  int iter = 0;
  if (freqList.size() > 0) {
    for (; iter < (int)gsub.size(); iter++) {
      if (gsub[iter] >= freq) break;
      newFreq.push_back(freqList[iter]);
    }
  }

  vector<int> newList;
  for (int i = 0; i < (int)transactionList.size(); i++) 
    newList.push_back(transactionList[i]);

  vector<int> newnewList;
  for (int i = iter; i < (int)gsub.size(); i++) {
    int item = gsub[i];
    int freqCount = 0;

    for (int list = 0; list < (int)newList.size(); list++) {
      const Transaction &transaction = database.database[newList[list]];
      if (binary_search(transaction.itemsets.begin(), transaction.itemsets.end(), item) == true) {
	freqCount += 1;
	newnewList.push_back(newList[list]);
      }
    }
    newFreq.push_back(freqCount);
    newList = newnewList;
    newnewList.clear();
  }
}

/*************************************************************************
 * Update Transaction List
 *************************************************************************/
inline void Lcm::UpdateTransactionList(const Database &database, const vector<int> &transactionList, const vector<int> &labelledTransactionList, const vector<int> &q_sets, int item, vector<int> &newTransactionList, vector<int> &newLabelledTransactionList) {
  for (int i = 0; i < (int)transactionList.size(); i++) {
    const Transaction &transaction = database.database[transactionList[i]];
    int iter;
    for (iter = 0; iter < (int)q_sets.size(); iter++) {
      int q = q_sets[iter];
      if (q >= item) {
	if (binary_search(transaction.itemsets.begin(), transaction.itemsets.end(), q) == false) 
	  break;
      }
    }
    if (iter == (int)q_sets.size()){
      newTransactionList.push_back(transactionList[i]);
      if(transaction.label){
        newLabelledTransactionList.push_back(transactionList[i]);
      }
    }
  }
}

/***************************************************************************
 * Print itemsets
 ***************************************************************************/
inline void Lcm::PrintItemsets(const vector<int> &itemsets, const OccurenceDeriver &occ) {
   if ((int)itemsets.size() > 0) {
     for (int i = 0; i < (int)itemsets.size(); i++)
       out << itemsets[i] << " ";
     out << endl;
   }
}

/*****************************************************************************
 * calculrate core_i
 *****************************************************************************/
inline int Lcm::CalcurateCoreI(const vector<int> &itemsets, const vector<int> &freqList) {
  if (itemsets.size() > 0) {
    int current = freqList[freqList.size() - 1];
    for (int i = (int)freqList.size() - 2; i >= 0; i--) {
      if (current != freqList[i]) return itemsets[i+1];
    }
    return itemsets[0];
  }
  else 
    return 0;
}


/**************************************************************************
 * Prefix Preseaving Test 
 * Test whether p(i-1) is equal to q(i-1) or not
 **************************************************************************/
inline bool Lcm::PpcTest(const Database &database, vector<int> &itemsets, vector<int> &transactionList, int item, vector<int> &newTransactionList) {
  // j_sets: set not including items which are included in itemsets.
  // make transactionList pointing to the indexes of database including P \cup item
  CalcTransactionList(database, transactionList, item, newTransactionList);

  // check j s.t j < i, j \notin P(i-1) is included in every transaction of T(P \cup {i})
  for (vector<int>::iterator j = totalItem.begin(); *j < item; j++) {
    if (binary_search(itemsets.begin(), itemsets.end(), *j) == false &&
      CheckItemInclusion(database, newTransactionList, *j) == true) {
      return false;
    }
  }
  return true;
}

/****************************************************************************
 * Make closure 
 * make Q = Clo(P \cup {i}) subject to Q(i-1) = P(i-1)
 *****************************************************************************/
inline void Lcm::MakeClosure(const Database &database, vector<int> &transactionList, vector<int> &q_sets, vector<int> &itemsets, int item) {
  for (int i = 0; i < (int)itemsets.size() && itemsets[i] < item; i++) {
    q_sets.push_back(itemsets[i]);
  }
  q_sets.push_back(item);

  vector<int>::iterator i = lower_bound(totalItem.begin(), totalItem.end(), item + 1);

  for (int iter = *i; i != totalItem.end(); i++, iter = *i) {
    if (CheckItemInclusion(database, transactionList, iter) == true) {
      q_sets.push_back(iter);
    }
  }
}

/********************************************************************************
 * CheckItemInclusion
 * Check whther item is included in the transactions pointed to transactionList
 ********************************************************************************/
inline bool Lcm::CheckItemInclusion(const Database &database, vector<int> &transactionList, int item) {
  for (vector<int>::iterator iter = transactionList.begin(); iter != transactionList.end(); iter++) {
    const Transaction &transaction = database.database[*iter];
    if (binary_search(transaction.itemsets.begin(), transaction.itemsets.end(), item) == false) return false;
  }
  return true;
}


/*********************************************************************************
 *  Calcurate new transaction list
 *********************************************************************************/
inline void Lcm::CalcTransactionList(const Database &database, const vector<int> &transactionList, int item, vector<int> &newTransactionList) {

  for (int list = 0; list < (int)transactionList.size(); list++) {
    const Transaction &transaction = database.database[transactionList[list]];
    if (binary_search(transaction.itemsets.begin(), transaction.itemsets.end(), item) == true) 
      newTransactionList.push_back(transactionList[list]);
  }
}

/***********************************************************************************
 * Update Occurence Deriver
 ***********************************************************************************/
inline void Lcm::UpdateOccurenceDeriver(const Database &database, const vector<int> &transactionList, OccurenceDeriver &occurence) {
  occurence.Clear();
  for (int i = 0; i < (int)transactionList.size(); i++) {
    const Transaction &transaction = database.database[transactionList[i]];
    const vector<int> &itemsets = transaction.itemsets;
    for (int j = 0; j < (int)itemsets.size(); j++) {
      occurence.table[itemsets[j]].push_back(transactionList[i]);
      if(transaction.label){
        occurence.labelledOccurenceNums[itemsets[j]]++;
      }
    }
  }
}

map<vector<int>, pair<int, int> > Lcm::getFrequencies(Database database, const set<vector<int> > &patterns, const Trie &trie){
  map<vector<int>, pair<int, int> > freqs;

  OccurenceDeriver occ(database);
  maxItem = database.GetMaxItem();
  minItem = database.GetMinItem();

  vector<int> itemsets; 
  vector<int> transactionList;
  for (int i = 0; i < database.GetNumberOfTransaction(); i++) {
    transactionList.push_back(i);
  }
  
  totalItem = database.GetItemset();
  pair<int, int> curFreq = make_pair(database.GetNumberOfLabelledTransaction(), database.GetNumberOfTransaction());
  vector<int> freqList;
  getFrequenticesIter(database, patterns, itemsets, transactionList, occ, freqList, curFreq, freqs, trie);
  for (set<vector<int> >::iterator it=patterns.begin(); it!=patterns.end(); ++it){
    if(freqs.find(*it) == freqs.end()){
      freqs[*it] = make_pair(0, 0);
    }
  }
	
  return freqs;
}

void Lcm::getFrequenticesIter(Database &database, const set<vector<int> > &patterns,
	vector<int> &itemsets, vector<int> &transactionList, OccurenceDeriver &occ, vector<int> &freqList,
	pair<int,int> curFreq, map<vector<int>, pair<int,int> > &freqs, const Trie &trie) {
  
  if(!trie.exist(itemsets)){ //reached at the leaf
    return;
  }

  if( patterns.find(itemsets) != patterns.end() ){
    freqs[itemsets] = curFreq;
  }

  int core_i = CalcurateCoreI(itemsets, freqList);
  vector<int>::iterator iter = lower_bound(totalItem.begin(), totalItem.end(), core_i);

  vector<int> freq_i;
  map<int, pair<int, int> > freq_n;
  for (int i = *iter; iter != totalItem.end(); iter++, i = *iter) {
    if((int)occ.table[i].size() >= 1 && binary_search(itemsets.begin(), itemsets.end(), i) == false){
      freq_i.push_back(i);
      freq_n[i] = make_pair(occ.GetNumLabelledOcc(i), occ.GetNumOcc(i));
    }
  }
  vector<int> newTransactionList;
  vector<int> newLabelledTransactionList;
  vector<int> q_sets;
  vector<int> newFreqList;
  for (vector<int>::iterator freq = freq_i.begin(); freq != freq_i.end(); freq++) {
    newTransactionList.clear();
    newLabelledTransactionList.clear();
    if (PpcTest(database, itemsets, transactionList, *freq, newTransactionList)) {
      q_sets = vector<int>(itemsets);
      q_sets.push_back(*freq);
      if (max_pat == 0 || (int)q_sets.size() <= max_pat) {
        newTransactionList.clear();
        newLabelledTransactionList.clear();
        UpdateTransactionList(database, transactionList, newTransactionList, q_sets, *freq, newTransactionList, newLabelledTransactionList);
        UpdateOccurenceDeriver(database, newTransactionList, occ);
        newFreqList.clear();
        UpdateFreqList(database, transactionList, q_sets, freqList, *freq, newFreqList);
        getFrequenticesIter(database, patterns, q_sets, newTransactionList, occ, newFreqList, freq_n[*freq], freqs, trie);
      }
    }
  }
}

