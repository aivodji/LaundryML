#ifndef _MHT_H_
#define _MHT_H_

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <ctime>
#include "Database.h"
#include "OccurenceDeriver.h"
#include "Lcm.h"
#include "tree.h"

#define MHT_MHT 0

using namespace std;

class Mht {
  ostream &out;
public:
  bool mode;     // mode = 0 (BH), 1 (BY)
  //double correction;     // m_0/m correction
  Mht(ostream &out, bool mode);
  ~Mht();
  int nextMinSupCurrent(int min_sup_min, int min_sup_max);
  void RunMhtFWER(Database &database, string outfile, int max_pat, double q, double a);
  void RunMht(Database &database, string outfile, int max_pat, double q, double a, Database &trainingDatabase, int fixedtau = 0);
  void RunEP(Database &database, string outfile, int min_sup, int max_pat, double a, double q); //standard EP mining 
  void RunNaiveEP(Database &database, string outfile, int max_pat, double a, double q); //Naive BH/BY
  vector<int> GetRejectedHypothesesExcludeParents(const vector<double> &pvalues, double q, double correction, const vector<vector<int> > &patterns);
};

#endif // _MHT_H_
