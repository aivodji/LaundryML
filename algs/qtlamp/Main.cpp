#include <iostream>
#include <ostream>
#include <cstring>
#include <string>
#include <vector>
#include "Lcm.h"
#include "Mht.h"
#include "Database.h"

/* Globals */
void Usage();
void ParseParameters (int argc, char **argv);

vector<string> filenames;
string trainingFilename;
bool           toggle_verbose       = false;
int            min_sup              = 1;    // minimum support for LCM
int            max_pat              = 0;    // maximum_itemset_size : default infinity;
double delta = 0.05;
double wr = 0.5;
string mining_mode;
string outfile;
bool use_by = false;

using namespace std;

/*****************************************************************************************
 * The main function
 *****************************************************************************************/
int main(int argc, char **argv) 
{
  ParseParameters(argc, argv);

  // set dataset
  if (toggle_verbose) {
    cerr << "set database" << endl;
    cerr << "filename:" << filenames[0] << endl;
    cerr << "filename_plus:" << filenames[1] << endl;
  }
  Database database;
  //Database database_cp;
  //Database database_plus;
  database.ReadFileAddLabel(filenames[0]);
  //database_cp.ReadFile(filenames[0]);
  //database_plus.ReadFile(filenames[1]);
  if( database.GetMaxItem() > ITEM_MAX ){
    cerr << "Error: MaxItem exceeds " << ITEM_MAX << endl;
    exit(0);
  }

  // set LCM parameter
  //Lcm lcm(cout, min_sup, max_pat);
  //lcm.RunLcm(database);
  Mht mht(cout, use_by);
  cout << "MHT initialized: min_sup=" << min_sup << " max_pat=" << max_pat << " delta=" << delta << " wr=" << wr << " out=" << outfile << endl;
  cout << "mining mode = " << mining_mode << " use_by=" << use_by << endl;
  if("W" == mining_mode){
    mht.RunMhtFWER(database, outfile, max_pat, delta, wr);
  }else if("R" == mining_mode){
    Database trainingDatabase;
    trainingDatabase.ReadFileAddLabel(trainingFilename);
    mht.RunMht(database, outfile, max_pat, delta, wr, trainingDatabase);
  }else if("E" == mining_mode){
    mht.RunEP(database, outfile, min_sup, max_pat, wr, delta);
  }else if("F" == mining_mode){ //FDR mining with fixed value of tau
    Database trainingDatabase;
    trainingDatabase.ReadFileAddLabel(trainingFilename);
    mht.RunMht(database, outfile, max_pat, delta, wr, trainingDatabase, min_sup);
  }else if("N" == mining_mode){ //FDR mining with 2^I hypotheses
    mht.RunNaiveEP(database, outfile, max_pat, wr, delta);
  }else if("L" == mining_mode){ //LCM
    Lcm lcm(cout, max_pat);
    const vector<vector<int> > patterns = lcm.RunLcm(database, min_sup, false).first;
    if(outfile.length() > 0){
      cout << "saving patterns to file " << outfile << endl;
      ofstream ofs( outfile.c_str() );
      for(int i=0;i<patterns.size();++i){
        string str = patternToString(patterns[i]);
       ofs << str << endl;
      } 
      ofs.close();
    }
    cout << "min_sup=" << min_sup << " patternnum=" << patterns.size() << endl;
  }else{
    cout << "unknown mining mode" << endl;
  }

  return 0;
}

/***************************************************************************
 * Usage
 ***************************************************************************/
void Usage(){
  cerr << endl
       << "Usage: lcm [OPTION]... (W/R/E/F/N/L) INFILE OUTFILE" << endl << endl
       << "       where [OPTION]...  is a list of zero or more optional arguments" << endl
       << "             W: fWer  R:fdR  E:Emerging (no stats) " << endl
       << "             INFILE(s)    is the name of the input transaction database" << endl << endl
       << "Additional arguments (at most one input file may be specified):" << endl
       << "       -min_sup [minimum support]" << endl
       << "       -max_pat [maximum pattern]" << endl
       << "Additional arguments:" << endl
       << "       -delta [false discovery rate (default 0.05)]" << endl
       << "       -wr [winning rate of interest (default 0.5)]" << endl
       << "       -of [output filename]" << endl
       << endl;
  exit(0);
}

/*****************************************************************************
 * ParseParameters
 *****************************************************************************/
void ParseParameters (int argc, char **argv){
  if (argc == 1) Usage();
  filenames.clear();

  for (int argno = 1; argno < argc; argno++){
    if (argv[argno][0] == '-'){
      if (!strcmp (argv[argno], "-verbose")) {
	toggle_verbose = true;
      }
      else if (!strcmp (argv[argno], "-min_sup")) {
	if (argno == argc - 1) cerr << "Must specify minimum support after -min_sup" << endl;
	min_sup = atoi(argv[++argno]);
      }
      else if (!strcmp (argv[argno], "-max_pat")) {
	if (argno == argc - 1) cerr << "Must specify miximum itemset size after -max_size" << endl;
	max_pat = atoi(argv[++argno]);
      }
      else if (!strcmp (argv[argno], "-delta")) {
	delta = atof(argv[++argno]);
      }
      else if (!strcmp (argv[argno], "-wr")) {
	wr = atof(argv[++argno]);
      }
      else if (!strcmp (argv[argno], "-by")) {
	use_by = true;
      }
      else if (!strcmp (argv[argno], "-of")) {
	outfile = argv[++argno];
      }
      else if (!strcmp (argv[argno], "-training")) {
	if (argno == argc - 1) cerr << "Must specify a filename after -training" << endl;
	trainingFilename = string(argv[++argno]);
      }
    }
    else if(0 == mining_mode.length()){
      mining_mode = argv[argno]; 
    } else {
      filenames.push_back(argv[argno]);
    }
  }
}
