# LaundryML


**LaundryML is a regularized model enumeration algorithm that enumerates rule lists for rationalization.

This code is based on the implementation of the Certifiably Optimal RulE ListS (CORELS) algorithm. For information on CORELS, please visit [its website](https://corels.eecs.harvard.edu).

### Python dependencies
* pip install -r requirements.txt

### C/C++ dependencies

* [gmp](https://gmplib.org/) (GNU Multiple Precision Arithmetic Library)
* [mpfr](http://www.mpfr.org/) (GNU MPFR Library for multiple-precision floating-point computations; depends on gmp)
* [libmpc](http://www.multiprecision.org/) (GNU MPC for arbitrarily high precision and correct rounding; depends on gmp and mpfr)

### Compilation
* make

### Scripts for model rationalization 

* [Adult Income]: python benchmark_adult.py
* [ProPublica Recidivism]: python benchmark_compas.py
* Result will be generated in the folder "res"

### Scripts for outcome rationalization 

* [Adult Income]: python benchmark_local_adult.py
* [ProPublica Recidivism]: python benchmark_local_compas.py
* Result will be generated in the folder "res_local"


### Visualizing results for model rationalization 

#### Models performances
* First run python summary_rationalization_both.py
* Run Rscript both_analysis.R in the "plotting_scripts" repository
* Results are saved in plotting_scripts/data/graphs
#### Audit of best models
* First run python selection_adult.py and python selection_compas.py to get the best models='s id, beta and lambda
* Run python summary_audit_adult.py and python summary_compas_adult.py with the corresponding pareameters
* Results are saved in plotting_scripts/data/graphs


### Visualizing results for outcome rationalization 

#### Models performances
* Run Rscript analysis_local_adult.R and Rscript analysis_local_adult.R in the "plotting_scripts" repository
* Results are saved in plotting_scripts/data/graphs