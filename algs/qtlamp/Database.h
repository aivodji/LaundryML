#ifndef _TRANSACTION_H_
#define _TRANSACTION_H_

#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <fstream>
#include <strstream>
#include <algorithm>

#include "OccurenceDeriver.h"

using namespace std;

class OccurenceDeriver;

struct Transaction {
  int id;               // transaction id
  vector<int> itemsets; // list of items
  set<int> prefix;
  int label;
  
  void clear() {
    id = 0;
    label = 0;
    itemsets.clear();
  }
};

class Database {
  int max_item;                 // max item;
  int min_item;                 // min item;
  int labelledTransactionNum;
public:
  vector<Transaction> database;            // transaction database
  vector<vector<unsigned short> > bitmap;  // bitmap representation
//  map<vector<int>, vector<int> > occurrences; //itemset -> occurrence list
//  map<vector<int>, vector<int> > labelledOccurrences; 

  Database();
  ~Database();
  int GetMaxItem(){ return max_item; }
  void Reduction(vector<int> &transactionList, int core_i);
  void RemoveItemsbyFreq(vector<int> &transactionList, OccurenceDeriver &occ, int core_i, int min_sup);
  void RemoveItemsbyFreq(int min_sup);
  void RemoveItems(vector<int> &itemset, vector<int> &transactionList, int core_i);
  vector<int> CalcurateIsuffList(vector<int> &transactionList, int core_i);
  void RemovePrefix(vector<int> &transactionList, int core_i);
//  void ReadFile(const string &filename);
  void ReadFileAddLabel(const string &filename);
  void FindMaxAndMinItem();
  vector<int> GetItemset();
  int GetId(int i) const;
  int GetItem(int i, int j) const;
  Transaction GetTransaction(int i) const;
  int GetMaxItem() const;
  int GetMinItem() const;
  int GetNumberOfTransaction() const;
  int GetNumberOfUniqueTransaction(int max_pat = 0) const;
  int GetNumberOfLabelledTransaction() const;
  unsigned short GetBitmap(int i, int item);
  void MakeBitmap();
//  void NormalizeLabels();
  void Print(ostream &out);
  void PrintBitmap(ostream &out);
//  void addOccurrences(const vector<int> &itemset, const vector<int> &os, const vector<int> &los);
};

#endif // _TRANSACTION_H_
