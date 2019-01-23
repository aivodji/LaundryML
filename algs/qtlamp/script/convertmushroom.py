#!/usr/bin/env python
# coding:utf-8

import random, sys

if __name__ == "__main__":
  lines = open(sys.argv[1], "r").readlines()
  attributes = set()
  for line in lines:
    sps = line.split(",")[1:]
    for i,sp in enumerate(sps):
      attributes.add((i,sp))
  attributes = sorted(list(attributes))
  attribute_to_featurenum = {}
  for i,attribute in enumerate(attributes):
    attribute_to_featurenum[attribute] = i+1
  for line in lines:
    label = 0
    if line[0]=="p":
      label = 1
    sps = line.split(",")[1:]
    features = []
    for i,sp in enumerate(sps):
      features.append(attribute_to_featurenum[(i,sp)])
    print " ".join(map(str, features))+ " " + str(label)
 
    
