#!/usr/bin/env python
# coding:utf-8

import random, sys

#adult dataset: a1a in libsvm

percentiles = [0.5, 1.0]

PRINT_TEMP = False

if __name__ == "__main__":
  lines = [line.strip() for line in open(sys.argv[1], "r").readlines()]
  attributes = set()
  feature_num = 21
  val_line = [[] for i in range(feature_num)]
  percentile_vals = [[] for i in range(feature_num)]
  for line in lines:
    sps = line.split(" ")[1:]
    for i,sp in enumerate(sps):
      cat = int(sp.split(":")[0])
      if cat <= feature_num:
        val = sp.split(":")[-1]
        val_line[cat-1].append(float(val))
  for i in range(feature_num):
    if PRINT_TEMP:
      print "feature",i+1
    val_line[i] = sorted(val_line[i])
    if PRINT_TEMP:
      print "min=",val_line[i][0]
      print "max=",val_line[i][-1]
    for percentile in percentiles:
      if PRINT_TEMP:
        print "percentile",percentile,"=",val_line[i][int(len(val_line[i])*percentile)-1]
      percentile_vals[i].append( val_line[i][int(len(val_line[i])*percentile)-1] )
  for line in lines:
    label = 0
    if line[0]!="-":
      label = 1
    sps = line.split(" ")[1:]
    features = []
    for i,sp in enumerate(sps):
      cat = int(sp.split(":")[0])
      val = sp.split(":")[-1]
      if cat <= feature_num:
        #print i,sp
        for j in range(len(percentiles)):
          if float(val) <= percentile_vals[cat-1][j]:
            features.append(1+(cat-1)*len(percentiles)+j)
            break
    print " ".join(map(str, features))+ " " + str(label)
 
    
