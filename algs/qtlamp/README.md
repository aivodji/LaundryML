QTLAMP: A library for Emerging Pattern Mining with Multiple Hypothesis Testing Correction

- - -
1\. [About](#about)  
2\. [Environment](#environment)  
3\. [Quick run](#quick)  
4\. [Misc](#misc)  
- - -

.
<a name="about"></a>

## 1\. About
  This is a C++ package for statistical emerging pattern mining (SEPM) simulations.
  For the details of SEPM, see [our paper](http://www.tkl.iis.u-tokyo.ac.jp/~jkomiyama/pdf/kdd-statistical-emerging.pdf) [1].
  The implementation of this package is based on [LCM++](https://code.google.com/archive/p/lcmplusplus/). This problem also uses [LCM](http://research.nii.ac.jp/~uno/code/lcm.html) as a component.
     
<a name="environment"></a>

## 2\. Environment
  This program supports a linux/GNU C++ environment. We do not check windows/MacOSX.
  
  Required packages:
  - C++0x: modern C++ compiler (preferably GNU C++ (g++))
  - libgmp: the GNU MP Bignum Library
  
<a name="quick"></a>

## 3\. Quick run
  Type 

    ./script/download.sh
    make
    time ./script/compare_methods.sh datasets/converted/svmguide3.label 0.3 10000000 6
    
  , which will finish within ten minutes with modern hardware. The result of the runs will be like
    
    qtlamp$ cat out/svmguide3-result-0.3-10000000-6-txt
    Result of file out/svmguide3-6.fdr.out
    Num of patterns found: 4971
    Result of file out/svmguide3-6.fdrby.out
    Num of patterns found: 1729
    Result of file out/svmguide3-6.fwer.out
    Num of patterns found: 536
    Result of file out/svmguide3-6.bh.out
    Num of patterns found: 1345
    Result of file out/svmguide3-6.ep.out
    Num of patterns found: 43461
  
  Note that the result of fdr minings can change slightly for each run due to its stochastic nature.
  
## References

    [1] Junpei Komiyama, Masakazu Ishihata, Hiroki Arimura, Takashi Nishibayashi, Shin-Ichi Minato. Statistical Emerging Pattern Mining with Multiple Testing Correction. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2017), Research Track, Hafifax, Nova Scotia, Canada, August 13-17, 2017 

## Author
  Junpei Komiyama (junpei.komiyama atmark gmail.com)
