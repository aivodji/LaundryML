#include "OccurenceDeriver.h"

/*************************************************************************
* Constructor
**************************************************************************/
OccurenceDeriver::OccurenceDeriver(Database &database) {
  tableSize = database.GetMaxItem() + 1;
  table.resize(tableSize);
  labelledOccurenceNums.resize(tableSize);

  int size = database.GetNumberOfTransaction();
  for (int i = 0; i < (int)size; i++) {
    const Transaction &transaction = database.GetTransaction(i);
    vector<int> itemsets = transaction.itemsets;
    for (vector<int>::iterator iter = itemsets.begin(); iter != itemsets.end(); iter++) {
      //		for (int j = 0; j < (int)itemsets.size(); j++) {
      table[*iter].push_back(i);
      if(transaction.label){
        labelledOccurenceNums[*iter]++;
      }
    }
  }
}

/***************************************************************************
* get the number of occurence of item
****************************************************************************/
int OccurenceDeriver::GetNumOcc(int item) const {
  return (int)table[item].size();
}

int OccurenceDeriver::GetNumLabelledOcc(int item) const {
  return labelledOccurenceNums[item];
}

/****************************************************************************
* clear table
*****************************************************************************/
void OccurenceDeriver::Clear() {
  for (int i = 0; i < (int)table.size(); i++){
    table[i].clear();
    labelledOccurenceNums[i] = 0;
  }
}

/******************************************************************************
* Print Occurence Deriver
*******************************************************************************/
void OccurenceDeriver::Print(ostream &out) {
  for (int i = 0; i < (int)table.size(); i++) {
    out << i << " : ";
    for (int j = 0; j < (int)table[i].size(); j++) {
      cout << table[i][j] << " " << endl;
    }
  }
}

/********************************************************************************
 * Push
 *******************************************************************************/
void OccurenceDeriver::Push(int item, int id) {
  table[item].push_back(id);
}

/*********************************************************************************
 * GetTable
 *********************************************************************************/
vector<int> OccurenceDeriver::GetTable(int item) const {
  return table[item];
}
