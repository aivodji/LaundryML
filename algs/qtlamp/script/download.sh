#!/bin/bash
cd lcm
echo "downloading lcm."
wget --no-check-certificate http://research.nii.ac.jp/~uno/code/lcm53.zip
unzip lcm53.zip
echo "compiling lcm."
make
cd ..
cd datasets/original
echo "downloading datasets..."
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2
bunzip2 covtype.libsvm.binary.bz2
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2
bunzip2 ijcnn1.bz2
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/svmguide3
wget --no-check-certificate https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data
cd ../..
echo "converting datasets"
python2 script/convertadult.py datasets/original/a1a > datasets/converted/a1a.label
python2 script/convertcodrna.py datasets/original/cod-rna > datasets/converted/codrna.label
python2 script/convertcovtype.py datasets/original/covtype.libsvm.binary > datasets/converted/covtype.label
python2 script/convertijcnn.py datasets/original/ijcnn1 > datasets/converted/ijcnn1.label
python2 script/convertphishing.py datasets/original/phishing > datasets/converted/phishing.label
python2 script/convertsvmguide3.py datasets/original/svmguide3 > datasets/converted/svmguide3.label
python2 script/convertmushroom.py datasets/original/agaricus-lepiota.data > datasets/converted/mushroom.label
cd datasets/converted
wget --no-check-certificate https://www.dropbox.com/s/7ld9t5jym3kzqdq/splatoon.label?dl=0
mv splatoon.label?dl=0 splatoon.label
