#ifndef _TREE_H_
#define _TREE_H_

#include <iostream>
#include <sstream>
#include <utility>
#include <vector>
#include <map>
#include <random>
#include <functional> //hash
#include <unordered_set>
#include <set>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <gmpxx.h>

using namespace std;

//typedef unordered_set set;

#define P_HOEFFDING 0
#define P_CHERNOFF 1
#define P_EXACT 2
#define P_MODE P_EXACT

extern std::mt19937 randomEngine;

#define pi 3.14159
#define ITEM_MAX 10000

class PNode {
public:
	vector<bool> pattern;
	int HypNum; 
	int N;
	int Nplus;
	PNode(){
		HypNum=N=Nplus=-1;
	}
};

template< typename T >
class tree_node {
public:
	T data;
	vector<tree_node> children;
};

class Trie {
  bool isLeaf;
  vector<Trie *> next;
public:
  Trie() { isLeaf = true; }
  void insert(const vector<int> &pattern); 
  bool exist(const vector<int> &pattern) const ; 
  bool isPatternLeaf(const vector<int> &pattern) const ; 
};

mpf_class mpf_pow(mpf_class a, int n);

double kl(double p, double q);

double getPvalue(int nplus, int n, double a, int mode=P_MODE);
vector<double> GetPvalues(double a, const vector<vector<int> > &patterns, const map<vector<int>, pair<int, int> > &ns); 

int getMinimumN(double a, double p, int mode=P_MODE);

double inverseSumToM(int m);

vector<int> GetRejectedHypotheses(const vector<double> &pvalues, double q, bool mode, double correction); //BH
vector<int> GetRejectedHypothesesWithGivenM(const vector<double> &pvalues, double q, bool mode, double m); //BH
vector<int> GetRejectedHypothesesFWER(const vector<double> &pvalues, double q); //Bonfelloni

string patternToString(const vector<int> &pattern);
string itos(int number);
string dtos(double number);

//nCk (combination), approximated
double nCkStirling(int n, int k);

#endif // _TREE_H_
