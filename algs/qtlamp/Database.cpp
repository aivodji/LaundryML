#include "Database.h"

/***************************************************************
* constructor
****************************************************************/
Database::Database() {}

/***************************************************************
* destructor
****************************************************************/
Database::~Database() {}

/***************************************************************
 * Database Reduction
 ***************************************************************/
void Database::Reduction(vector<int> &transactionList, int core_i) {
  RemovePrefix(transactionList, core_i);
}

/***************************************************************
 * Remove infrequent Items from database
 **************************************************************/
inline void Database::RemoveItemsbyFreq(vector<int> &transactionList, OccurenceDeriver &occ, int core_i, int min_sup) {
  for (int item = min_item; item < core_i; item++) {
    if (occ.GetNumOcc(item) < min_sup) {
      for (int list = 0; list < (int)transactionList.size(); list++) {
	Transaction &transaction = database[transactionList[list]];
	remove(transaction.itemsets.begin(), transaction.itemsets.end(), item);
      }
    }
  }
}

/****************************************************************
 * Remove items inclued in itemset from database
 ****************************************************************/
inline void Database::RemoveItems(vector<int> &itemset, vector<int> &transactionList, int core_i) {
  for (int iter = 0; iter < (int)itemset.size() && itemset[iter] < core_i; iter++) {
    int item = itemset[iter];
    for (int list = 0; list < (int)transactionList.size(); list++) {
      Transaction &transaction = database[transactionList[list]];
      remove(transaction.itemsets.begin(), transaction.itemsets.end(), item);
    }
  }
}

/***************************************************************
 * Remove infrequent items from database
 **************************************************************/
void Database::RemoveItemsbyFreq(int min_sup) {
  OccurenceDeriver occ(*this);
  for (int item = min_item; item <= max_item; item++) {
    if (occ.GetNumOcc(item) < min_sup) {
      for (int list = 0; list < (int)database.size(); list++) {
	Transaction &transaction = database[list];
	remove(transaction.itemsets.begin(), transaction.itemsets.end(), item);
      }
    }
  }
  FindMaxAndMinItem();
}

/*****************************************************************
 * Remove prefixies that have the same saffix
 *****************************************************************/
inline void Database::RemovePrefix(vector<int> &transactionList, int core_i) {
  const vector<int> iSuffList = CalcurateIsuffList(transactionList, core_i);
  for (int item = min_item; item < core_i; item++) {
    bool flag = true;
    for (int list = 0; list < (int)iSuffList.size(); list++) {
      Transaction &transaction = database[iSuffList[list]];
      int iter = 0;
      for (; iter < (int)transaction.itemsets.size(); iter++) {
	if (transaction.itemsets[iter] == item) break;
      }
      if (iter == (int)transaction.itemsets.size()) {
	flag = false;
	break;
      }
    }
    if (flag == false) {
      for (int list = 0; list < (int)iSuffList.size(); list++) {
	Transaction &transaction = database[iSuffList[list]];
	for (vector<int>::iterator iter = transaction.itemsets.begin(); iter != transaction.itemsets.end(); iter++) {
	  if (*iter == item) {
	    transaction.itemsets.erase(iter);
	    break;
	  }
	}
      }
    }
  }
}

/***************************************************************************************
 * calcurate i_suffix
 ***************************************************************************************/
inline vector<int> Database::CalcurateIsuffList(vector<int> &transactionList, int core_i) {
  vector<int> iSuffList = transactionList;

  vector<int> delList;
  for (int item = max_item; item >= (int)core_i; item--) {
    vector<int> newList;
    for (int list = 0; list < (int)iSuffList.size(); list++) {
      const Transaction &transaction = database[iSuffList[list]];

      if (binary_search(transaction.itemsets.begin(), transaction.itemsets.end(), item) == true) {
	newList.push_back(iSuffList[list]);
      }
    }
    if (newList.size() != 0) {
      iSuffList = newList;
      newList.clear();
    }
  }

  return iSuffList;
}

/****************************************************************
* read file from filename
*****************************************************************/
/*
void Database::ReadFile(const string &filename) {
  string line;
  Transaction transaction;
  int item;
  int total_id = 0;
  cout << "reading file " << filename << endl;
  ifstream is(filename.c_str());
  while (getline (is, line)) {
    transaction.clear();
    istrstream istrs ((char *)line.c_str());
    while (istrs >> item) transaction.itemsets.push_back(item);
    sort(transaction.itemsets.begin(), transaction.itemsets.end());
    transaction.id = total_id++;
    database.push_back(transaction);
  }
  FindMaxAndMinItem();
}
*/

/*****************************************************************
* read file and label from filename
******************************************************************/
void Database::ReadFileAddLabel(const string &filename) {
  string line;
  Transaction transaction;
  int item;
  int total_id = 0;
  labelledTransactionNum = 0;
  cout << "reading file " << filename << endl;
  ifstream is(filename.c_str());
  while (getline (is, line)) {
    transaction.clear();
    istrstream istrs ((char *)line.c_str());
    while (istrs >> item) transaction.itemsets.push_back(item);
    transaction.id = total_id++;
    int label = transaction.itemsets.back();
    if(label !=0 && label != 1){
      cerr << "transaction label must be binary" << endl;exit(0);
    }
    if(label == 1){
      labelledTransactionNum++;
    }
    transaction.label = label; 
    transaction.itemsets.pop_back();
    sort(transaction.itemsets.begin(), transaction.itemsets.end());
    database.push_back(transaction);
  }
  FindMaxAndMinItem();
}

/*******************************************************************
* find max and min item from database
********************************************************************/
void Database::FindMaxAndMinItem() {
  max_item = 0;
  min_item = 100000000;
  for (int i = 0; i < (int)database.size(); i++) {
    Transaction &transaction = database[i];
    vector<int> &itemsets = transaction.itemsets;
    for (int j = 0; j < (int)itemsets.size(); j++) {
      if (itemsets[j] > max_item) max_item = itemsets[j];
      if (itemsets[j] < min_item) min_item = itemsets[j];
    }
  }
}

/*******************************************************************
* GetItemset()
********************************************************************/
vector<int> Database::GetItemset() {
    set<int> totalItem;
    for (int i = 0; i < (int)database.size(); i++) {
	Transaction &transaction = database[i];
	vector<int> &itemsets = transaction.itemsets;
	for (int j = 0; j < (int)itemsets.size(); j++) {
	    totalItem.insert(itemsets[j]);
	}
    }
    vector<int> allItem;
    for (set<int>::iterator iter = totalItem.begin(); iter != totalItem.end(); iter++) {
      allItem.push_back(*iter);
    }
    return allItem;
}

/*********************************************************************
* get ID of transaction
**********************************************************************/
int Database::GetId(int i) const {
  return database[i].id;
}

/**********************************************************************
*  get max_item
***********************************************************************/
int Database::GetMaxItem() const {
  return max_item;
}

/***********************************************************************
* get min_item
************************************************************************/
int Database::GetMinItem() const {
  return min_item;
}


/*************************************************************************
* get transaction
**************************************************************************/
Transaction Database::GetTransaction(int i) const {
  return database[i];
}

/************************************************************************
* get the number of transactions in database
*************************************************************************/
int Database::GetNumberOfTransaction() const {
  return (int)database.size();
}

int Database::GetNumberOfUniqueTransaction(int max_pat) const {
  set<vector<int> > unique_itemset;
  for(int i=0;i<database.size();++i){
    const vector<int> itemset = database[i].itemsets;
    if(max_pat == 0 || itemset.size() <= max_pat){
      unique_itemset.insert(itemset);
    }
  }
  return unique_itemset.size();
}

int Database::GetNumberOfLabelledTransaction() const {
  return labelledTransactionNum;
}

/************************************************************************
* print database
*************************************************************************/
void Database::Print(ostream &out) {
	out << "ID itemsets" << endl;
	for (int i = 0; i < (int)database.size(); i++) {
		Transaction &transaction = database[i];
		vector<int> &itemsets = transaction.itemsets;
		out << transaction.id << " ";
		for (int j = 0; j < (int)itemsets.size(); j++) {
			out << itemsets[j] << " ";
		}
		out << endl;
	}
	out << "max:" << max_item << " " << "min:" << min_item << endl;
}


/**************************************************************************
* normalize labels
* \sum_i y_i = 0
***************************************************************************/
/*
void Database::NormalizeLabels() {
  double sum = 0;
  for (int i = 0; i < (int) database.size(); i++) {
    sum += database[i].label;
  }
 
  for (int i = 0; i < (int) database.size(); i++) {
    database[i].label = database[i].label - (double)sum/(double)(database.size());
  }
}
*/

/**************************************************************************
 * MakeBitmap
 **************************************************************************/
void Database::MakeBitmap() {
  bitmap.resize(database.size());

  for (int i = 0; i < (int)database.size(); i++) {
    const Transaction &transaction = database[i];
    const vector<int> &itemset = transaction.itemsets;
    bitmap[i].resize(max_item - min_item + 1, 0);
    for (int j = 0; j < (int)itemset.size(); j++) {
      bitmap[i][itemset[j]-min_item] = 1;
    }
  }
}

/**************************************************************************
 * MakeBitmap
 **************************************************************************/
void Database::PrintBitmap(ostream &out) {
  for (int i = 0; i < (int)bitmap.size(); i++) {
    for (int j = 0; j < (int)bitmap[i].size(); j++) {
      out << bitmap[i][j] << " ";
    }
    out << endl;
  }
}

/**************************************************************************
 * GetBitmap
 **************************************************************************/
unsigned short Database::GetBitmap(int i, int item) {
  return bitmap[i][item];
}

/*
void Database::addOccurrences(const vector<int> &itemset, const vector<int> &os, const vector<int> &los){
  occurrences[itemset] = os;
  labelledOccurrences[itemset] = los;
}
*/
