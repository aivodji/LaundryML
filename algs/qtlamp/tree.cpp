#include "tree.h"

std::mt19937 randomEngine;

void Trie::insert(const vector<int> &pattern){
  Trie *r = this;
  for (int i = 0; i<pattern.size(); ++i) {
    int c = pattern.at(i);
    if(r->isLeaf){
      r->next.resize(ITEM_MAX);
      r->isLeaf = false;
    }
    if (!r->next[c]){
      r->next[c] = new Trie;
    }
    r = r->next[c];
  }
  return;
}
bool Trie::exist(const vector<int> &pattern) const {
  const Trie *r = this;
  for (int i = 0; i<pattern.size(); ++i) {
    int c = pattern.at(i);
    if(r->isLeaf){
      return false;
    }
    if (!r->next[c]){
      return false;
    }
    r = r->next[c];
  }
  return true;
}
bool Trie::isPatternLeaf(const vector<int> &pattern) const {
  const Trie *r = this;
  for (int i = 0; i<pattern.size(); ++i) {
    int c = pattern.at(i);
    if(r->isLeaf){
      cerr << "isLeaf points non-exist node" << endl;exit(0);
      return false;
    }
    if (!r->next[c]){
      cerr << "isLeaf points non-exist node" << endl;exit(0);
      return false;
    }
    r = r->next[c];
  }
  return r->isLeaf;
}

double kl(double p, double q){
  if(p>1 || p<0){
    cerr << "p = " << p << " is out of range [0,1]" << endl;exit(0);
  }
  if(q>=1 || q<=0){
    cerr << "q = " << q << " is out of range (0,1)" << endl;exit(0);
  }
  double v = 0;
  if(p>0) v += p*log(p/q);
  if(p<1) v += (1-p)*log((1-p)/(1-q));
  return v;
}

mpf_class mpf_pow(mpf_class a, int n){
  mpf_class p(1., 10000);
  for(int i=0;i<n;++i){
    p *= a;
  }
  return p;
}

double getPvalue(int nplus, int n, double a, int mode){ //Pr(n+ >= nplus) among n samples w/ positive prob a \in (0,1)
  if(n==0) return 1.;
  double mu = nplus/(double)n;
  if(mu <= a ) return 1.;
  if(mode == P_HOEFFDING){
    return exp( -2.*n*(mu-a)*(mu-a) );
  }else if(mode == P_CHERNOFF){
    return exp( -n*kl(mu, a) );
  }else if(mode == P_EXACT){
    if(n > 100){
      return exp( -n*kl(mu, a) );
    }else{
      vector<mpz_class> binomials(n+1, 0);
      binomials[0]=1;
      for(int i=0;i<n;++i){
        vector<mpz_class> binomials_tmp(n+1, 1);
        for(int j=1;j<=n;++j){
          binomials_tmp[j] = binomials[j]+binomials[j-1];
        }
        binomials = binomials_tmp;
      }
      mpf_class p(0, 10000);
      for(int t=n;t>=nplus;--t){
        p += mpf_pow(a, t) * mpf_pow(1.-a,n-t) * binomials[t];
      }
      return p.get_d();
    }
  }else{
    cerr << "Unknown pvalue mode" << endl;exit(0);
  }
}

vector<double> GetPvalues(double a, const vector<vector<int> > &patterns, const map<vector<int>, pair<int, int> > &ns){
  vector<double> pvalues;
  for(int i=0;i<patterns.size();++i){
    const int nplus = ns.at(patterns[i]).first, n = ns.at(patterns[i]).second;
    pvalues.push_back( getPvalue(nplus, n, a) );
    if(nplus > n){
      cerr << "Error: n+ > n for some pattern (" << nplus << "/" << n << "):";
      for(int l=0;l<patterns[i].size();++l){
        cerr << patterns[i][l] << "-";
      }
      cerr << endl;
      cerr << endl;exit(0);
    }
  }
  return pvalues;
}


int getMinimumN(double a, double p, int mode){
  if(a >= 1){
    cerr << "Error: a >= 1" << endl;
    exit(0);
  }
  if(p >= 1 || p <= 0){
    cerr << "Error: p is not in (0, 1)" << p << endl;
    exit(0);
  }
  if(mode == P_HOEFFDING){
    return int( log(1.0/p) / (2. * (1.-a) * (1.-a)) + 0.9999);
  }else if(mode == P_CHERNOFF){
    return int( log(1.0/p) / kl(1., a) + 0.9999);
  }else if(mode == P_EXACT){
    return int( log(p)/log(a) + 0.9999);
  }else{
    cerr << "Unknown pvalue mode" << endl;exit(0);
  }
}

// 1 + 1/2 + 1/3 + ... + 1/m, possibly slow
double inverseSumToM(int m){
  static map<int, double> isum;
  if( isum.find(m) != isum.end() ){
    return isum[m];
  }else{
    double s = 1;
    for(int i=2;i<=m;++i){
      s += 1.0/(double)i;
    }
    isum[m] = s;
    return s;
  }
}

vector<int> GetRejectedHypotheses(const vector<double> &pvalues, double q, bool mode, double correction){
  vector<pair<double, int> > ps_indices;
  vector<int> rejectedHypotheses;
  int i=0;
  for(vector<double>::const_iterator it=pvalues.begin();it!=pvalues.end();++it,++i){
    ps_indices.push_back(make_pair(*it,i));
  }
  sort(ps_indices.begin(),ps_indices.end());
  const int m = pvalues.size();
  for(i=0;i<m;++i){
    const int j=i+1;
    double threshold = j*q/(double)m;
    if(mode){
      threshold /= inverseSumToM(m*correction);
    }
    if(ps_indices[i].first > threshold){
      break;
    }else{
      rejectedHypotheses.push_back(ps_indices[i].second);
    }
  }
  return rejectedHypotheses;
}

vector<int> GetRejectedHypothesesWithGivenM(const vector<double> &pvalues, double q, bool mode, double m){
  vector<pair<double, int> > ps_indices;
  vector<int> rejectedHypotheses;
  int i=0;
  for(vector<double>::const_iterator it=pvalues.begin();it!=pvalues.end();++it,++i){
    ps_indices.push_back(make_pair(*it,i));
  }
  sort(ps_indices.begin(),ps_indices.end());
//  const int m = pvalues.size();
  for(i=0;i<pvalues.size();++i){
    const int j=i+1;
    double threshold = j*q/(double)m;
    if(mode){
      threshold /= inverseSumToM(m);
    }
    if(ps_indices[i].first > threshold){
      break;
    }else{
      rejectedHypotheses.push_back(ps_indices[i].second);
    }
  }
  return rejectedHypotheses;
}

vector<int> GetRejectedHypothesesFWER(const vector<double> &pvalues, double q){
  vector<int> rejectedHypotheses;
  for(unsigned int i=0;i<pvalues.size();++i){
    if(pvalues[i] <= q){
      rejectedHypotheses.push_back(i);
    }
  }
  return rejectedHypotheses;
}

string patternToString(const vector<int> &pattern){
  string astr; 
  for(unsigned int i=0;i<pattern.size();++i){
    astr += itos(pattern[i]);
    if(pattern.size() != (i+1)){
      astr += " ";
    }
  }
  return astr; 
}

string itos(int number){
  std::stringstream ss;
  ss << number;
  return ss.str();
}

string dtos(double number){
  std::stringstream ss;
  ss << number;
  return ss.str();
}

double nCkStirling(int N, int K) {
  double result = 1;
  int k;
  for(k = 1; k <= K; k++) {
    result *= (double)N/(double)k;
  }
  return result;
}
