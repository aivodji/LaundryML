#!/bin/bash
echo "script for batch mining - filename :" $1 " wr=" $2 " n=" $3 " max_pat=" $4
echo "generating data."
fname_ext="${1##*/}"
fname="${fname_ext%.*}"
python2 script/bootstrapping.py -f $1 -s -o temp/$fname.$4.split -n $3
echo "data generated."
echo "running QT-LAMP-EP(BH)."
./mht -of out/$fname-$4.fdr.out -wr $2 -max_pat $4 -delta 0.05 R ./temp/$fname.$4.split.1 -training ./temp/$fname.$4.split.2 > out/$fname-$4.fdr.stdout.txt
echo "running QT-LAMP-EP(BY)."
./mht -of out/$fname-$4.fdrby.out -wr $2 -max_pat $4 -delta 0.05 R ./temp/$fname.$4.split.1 -training ./temp/$fname.$4.split.2 -by > out/$fname-$4.fdrbystdout.txt
echo "running LAMP-EP."
./mht -of out/$fname-$4.fwer.out -wr $2 -max_pat $4 -delta 0.05 W ./temp/$fname.$4.split.0 > out/$fname-$4.fwerstdout.txt
echo "running Naive BH (naive FDR mining with all itemset)."
./mht -of out/$fname-$4.bh.out -wr $2 -max_pat $4 -delta 0.05 N ./temp/$fname.$4.split.0 > out/fname-$4.bhstdout.txt
echo "running standard EP."
./mht -of out/$fname-$4.ep.out -wr $2 -max_pat $4 -min_sup 10 E ./temp/$fname.$4.split.0 > out/$fname-$4.epbatch.txt
rm -f out/$fname-result-$2-$3-$4-txt
python2 script/bootstrapping.py -v -f out/$fname-$4.fdr.out -a out/$fname-result-$2-$3-$4-txt --wr $2
python2 script/bootstrapping.py -v -f out/$fname-$4.fdrby.out -a out/$fname-result-$2-$3-$4-txt --wr $2
python2 script/bootstrapping.py -v -f out/$fname-$4.fwer.out -a out/$fname-result-$2-$3-$4-txt --wr $2
python2 script/bootstrapping.py -v -f out/$fname-$4.bh.out -a out/$fname-result-$2-$3-$4-txt --wr $2
python2 script/bootstrapping.py -v -f out/$fname-$4.ep.out -a out/$fname-result-$2-$3-$4-txt --wr $2
