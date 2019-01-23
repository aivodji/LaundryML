#include "Mht.h"

/************************************************************************
 * Constructor
 ************************************************************************/
Mht::Mht(ostream &_out, bool mode) 
: out(_out), mode(mode) {}

/************************************************************************
 * Destructor
 ************************************************************************/
Mht::~Mht() {}

void Mht::RunMhtFWER(Database &database, string outfile, int max_pat, double q, double a){
  int min_sup_min = getMinimumN(a, q);
  cout << "min_sup_min =" << min_sup_min << endl;
  Lcm lcm(out, max_pat);
  vector<vector<int> > patterns;
  int min_sup_max = database.GetNumberOfTransaction(); 
  cout << "Starting binary search." << endl;

  while( min_sup_max > min_sup_min+1 ){ //binary search for finding tau(min_sup)

    const int min_sup_current = nextMinSupCurrent(min_sup_min, min_sup_max);
    cout << "min_sup_current =" << min_sup_current << endl;
    Database database_tmp(database);
    vector<vector<int> > newPatterns = lcm.RunLcm(database_tmp, min_sup_current, false).first;
    double current_reachable_q = pow(a, min_sup_current);
    const int new_m = newPatterns.size();
    double current_adjusted_q = q/new_m;
    if(current_reachable_q <= current_adjusted_q){
      min_sup_max = min_sup_current;
    }else{
      min_sup_min = min_sup_current;
    }
    patterns = newPatterns;
  }
  cout << "loop end: min_sup_max=" << min_sup_max << endl;
  const pair<vector<vector<int> >, Trie> tmp = lcm.RunLcm(database, min_sup_max, true);
  patterns = tmp.first;
  Trie trie = tmp.second;
  set<vector<int> > patterns_s;
  for(int i=0;i<patterns.size();++i){
    patterns_s.insert( patterns[i] );
  }
  map<vector<int>, pair<int, int> > ns = lcm.getFrequencies(database, patterns_s, trie);
  cout << "GetFreq - n+/n end." << endl;
  vector<double> pvalues = GetPvalues(a, patterns, ns);
  cout << "Pvalues calculated." << endl;

  vector<int> rejectedHypotheses = GetRejectedHypothesesFWER(pvalues, q/patterns.size());
  vector<vector<int> > outputPatterns;
  cout << "Listing statistically significant (FWER)" << rejectedHypotheses.size() << " patterns with q=" << q << ":" << endl;
  for(int i=0;i<rejectedHypotheses.size();++i){
    int index = rejectedHypotheses[i];
    vector<int> pat = patterns[index];
    cout << patternToString(pat) << " (" << pvalues[index] << ")" << endl;
    outputPatterns.push_back(pat);
  }
  cout << "marginal p-value: " << q/patterns.size() << endl;

  if(outfile.length() > 0){
    cout << "saving patterns to file " << outfile << endl;
    ofstream ofs( outfile.c_str() );
    for(int i=0;i<outputPatterns.size();++i){
      string str = patternToString(outputPatterns[i]);
      ofs << str << endl;
    } 
    ofs.close();
  }
}

int Mht::nextMinSupCurrent(int min_sup_min, int min_sup_max){
  int tmp_min = min_sup_min + 1;
  int tmp_max = min_sup_max - 1;
  int tmp = (1*min_sup_max+1*min_sup_min)/2;
  return min(max(tmp_min, tmp), tmp_max);
}

void Mht::RunMht(Database &mainDatabase, string outfile, int max_pat,
    double q, double a, Database &caribDatabase, int fixedtau){
  // set LCM parameter
  int min_sup_min = getMinimumN(a, q);
  cout << "min_sup_min =" << min_sup_min << endl;
  Lcm lcm(out, max_pat);
  vector<vector<int> > patterns;
  Trie trie;
  int min_sup_max = caribDatabase.GetNumberOfTransaction();
  if(0 == fixedtau){
    cout << "starting FDR binary search" << endl;
    int count = 0;
    while( min_sup_max > min_sup_min+1 ){ //binary search over min_sup (=tau)
      const int min_sup_current = nextMinSupCurrent(min_sup_min, min_sup_max);
      count++;
      cout << "count=" << count  << " current min_sup=" << min_sup_current << endl;
      
      const pair<vector<vector<int> >, Trie> tmp = lcm.RunLcm(caribDatabase, min_sup_current, true);
      vector<vector<int> > newTempPatterns = tmp.first; //H1
      const Trie trie_h1 = tmp.second;
      const int newM = newTempPatterns.size();

      set<vector<int> > newTempPatterns_s;
      for(int i=0;i<newTempPatterns.size();++i){
        newTempPatterns_s.insert( newTempPatterns[i] );
      }
      map<vector<int>, pair<int, int> > ns_tmp = lcm.getFrequencies(caribDatabase, newTempPatterns_s, trie_h1); 
      for (std::set<vector<int>>::iterator it=newTempPatterns_s.begin(); it!=newTempPatterns_s.end(); ++it){
        if( ns_tmp.find(*it) == ns_tmp.end() ){
          ns_tmp[*it] = make_pair(0,0);
        }
      }
      if(ns_tmp.size() != newTempPatterns_s.size()){
        cerr << "Error: pattern size " << newTempPatterns_s.size()<< " and frequency size " << ns_tmp.size() << " does not match!" << endl;
        exit(0);
      }
      vector<double> pvalues_tmp = GetPvalues(a, newTempPatterns, ns_tmp);
      vector<int> rejectedHypotheses = GetRejectedHypotheses(pvalues_tmp, q, mode, 1.0);
      const uint fdrest_k = max<int>(1, rejectedHypotheses.size());
      double adjusted_q = q * fdrest_k / (double)pvalues_tmp.size();
      if(mode){
        adjusted_q /= inverseSumToM(pvalues_tmp.size());
      }

      const int min_sup_adjusted = getMinimumN(a, adjusted_q); 

      if(min_sup_adjusted > min_sup_current){
        cout << "H1>H2" << endl;
        min_sup_min = min_sup_current;
      }else{
        cout << "H1<=H2" << endl;
        min_sup_max = min_sup_current;
      }

      cout << "iter end" << endl;
    }
  }else{ //fixedtau
    min_sup_max = fixedtau;
  }
  cout << "loop end: min_sup_max=" << min_sup_max << endl;
  const pair<vector<vector<int> >, Trie> tmp2 = lcm.RunLcm(mainDatabase, min_sup_max, true);
  patterns = tmp2.first;
  trie = tmp2.second;

  set<vector<int> > patterns_s;
  for(int i=0;i<patterns.size();++i){
    patterns_s.insert( patterns[i] );
  }
  cout << "patterns.size()=" << patterns.size() << endl;

  map<vector<int>, pair<int, int> > ns = lcm.getFrequencies(mainDatabase, patterns_s, trie);
  cout << "GetFreq - n+/n end. size=" << ns.size() << endl;

  vector<double> pvalues = GetPvalues(a, patterns, ns);
  vector<int> rejectedHypotheses = GetRejectedHypotheses(pvalues, q, mode, 1.0);
  vector<vector<int> > outputPatterns;
  cout << "Listing statistically significant " << rejectedHypotheses.size() << " patterns with q=" << q << ":" << endl;
  for(int i=0;i<rejectedHypotheses.size();++i){
    int index = rejectedHypotheses[i];
    vector<int> pat = patterns[index];
    cout << patternToString(pat) << " (" << pvalues[index] << ") N+/N=" << ns[pat].first << "/" << ns[pat].second << endl;
    outputPatterns.push_back(pat);
  }

  double marginal_pvalue = q * (rejectedHypotheses.size()+1) / patterns.size();
  if(mode){
    marginal_pvalue /= inverseSumToM(patterns.size());
  }
  cout << "marginal p-value: " << marginal_pvalue << endl;
  if(outfile.length() > 0){
    cout << "saving patterns to file " << outfile << endl;
    ofstream ofs( outfile.c_str() );
    for(int i=0;i<outputPatterns.size();++i){
      string str = patternToString(outputPatterns[i]);
      ofs << str << endl;
    } 
    ofs.close();
  }
}


void Mht::RunEP(Database &database, string outfile, int min_sup, int max_pat, double a, double q){
  Lcm lcm(out, max_pat);
  cout << "Running LCM with min_sup = " << min_sup << "." << endl;
  const pair<vector<vector<int> >, Trie> tmp = lcm.RunLcm(database, min_sup, true);
  const vector<vector<int> > patterns = tmp.first; 
  const Trie trie = tmp.second;
  cout << "LCM end." << patterns.size() << " patterns found" << endl;
  set<vector<int> > patterns_s;
  for(int i=0;i<patterns.size();++i){
    patterns_s.insert( patterns[i] );
  }
  map<vector<int>, pair<int, int> > ns = lcm.getFrequencies(database, patterns_s, trie);
  vector<vector<int> > EPPatterns;
  cout << "GetFreq - n+/n end." << endl;
  
  for(int i=0;i<patterns.size();++i){
    vector<int> pattern = patterns[i];
    if(!ns[pattern].second){
      cerr << "Error: pattern with zero support" << endl;
      exit(0);
    }
    if( ns[pattern].first/(double)(ns[pattern].second) > a ){
      EPPatterns.push_back(patterns[i]);
    }
  }
  cout << EPPatterns.size() << " emerning patterns found" << endl;

  if(outfile.length() > 0){
    cout << "saving patterns to file " << outfile << endl;
    ofstream ofs( outfile.c_str() );
    for(int i=0;i<EPPatterns.size();++i){
      string str = patternToString(EPPatterns[i]);
      ofs << str << endl;
    } 
    ofs.close();
  }
}

void Mht::RunNaiveEP(Database &database, string outfile, int max_pat, double a, double q){
  Lcm lcm(out, max_pat);
  int min_sup = getMinimumN(a, q);
  cout << "Running LCM with min_sup = " << min_sup << "." << endl;
  const pair<vector<vector<int> >, Trie> tmp = lcm.RunLcm(database, min_sup, true);
  const vector<vector<int> > patterns = tmp.first; //lcm.RunLcm(database, min_sup, false).first;
  const Trie trie = tmp.second;
  cout << "LCM end." << patterns.size() << " patterns found" << endl;
  set<vector<int> > patterns_s;
  for(int i=0;i<patterns.size();++i){
    patterns_s.insert( patterns[i] );
  }
  map<vector<int>, pair<int, int> > ns = lcm.getFrequencies(database, patterns_s, trie);
  vector<vector<int> > EPPatterns;
  cout << "GetFreq - n+/n end." << endl;
  
  vector<double> pvalues = GetPvalues(a, patterns, ns);
  cout << "max_item = " << database.GetMaxItem() << endl;
  double M;
  if(0 == max_pat){
    M = pow(2, database.GetMaxItem());
  }else{
    M = 0;
    int I = database.GetMaxItem();
    for(int k=0;k<=max_pat;++k){
      //binomial coef, stirling approximation
      //http://math.stackexchange.com/questions/1447296/stirlings-approximation-for-binomial-coefficient
      double I_C_k = nCkStirling(I, k);
      M += I_C_k;
    }
  }
  cout << "M=" << M << endl;
  vector<int> rejectedHypotheses = GetRejectedHypothesesWithGivenM(pvalues, q, mode, M);
  cout << "Listing statistically significant " << rejectedHypotheses.size() << " patterns with q=" << q << ":" << endl;
  for(int i=0;i<rejectedHypotheses.size();++i){
    int index = rejectedHypotheses[i];
    vector<int> pat = patterns[index];
    EPPatterns.push_back(pat);
  }

  double marginal_pvalue = q * (rejectedHypotheses.size()+1) / M;
  if(mode){
    marginal_pvalue /= inverseSumToM(patterns.size());
  }
  cout << "marginal p-value: " << marginal_pvalue << endl;

  if(outfile.length() > 0){
    cout << "saving patterns to file " << outfile << endl;
    ofstream ofs( outfile.c_str() );
    for(int i=0;i<EPPatterns.size();++i){
      string str = patternToString(EPPatterns[i]);
      ofs << str << endl;
    } 
    ofs.close();
  }
}
