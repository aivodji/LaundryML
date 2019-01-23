#!/usr/bin/env python
# coding:utf-8

mode = "phishing"
import random, sys

def isNeg(astr):
  return astr=="0"

def discretize(astr):
  #print astr
  return int(float(astr.split(":")[-1]))

if __name__ == "__main__":
  lines = open(sys.argv[1], "r").readlines()
  attributes = set()
  for line in lines:
    line = line.strip()
    sps = [sp for sp in line.split(" ")[1:] if len(sp)>0]
    for i,sp in enumerate(sps):
      attributes.add((i,sp))
  attributes = sorted(list(attributes))
  attribute_to_featurenum = {}
  for i,attribute in enumerate(attributes):
    attribute_to_featurenum[attribute] = i+1
  #print attribute_to_featurenum;sys.exit(0)
  for line in lines:
    line = line.strip()
    label = 1
    if isNeg(line.split(" ")[0]):
      label = 0
    sps = [sp for sp in line.split(" ")[1:] if len(sp)>0]
    features = []
    for i,sp in enumerate(sps):
      features.append(attribute_to_featurenum[(i,sp)])
    print " ".join(map(str, features))+ " " + str(label)
 
  
