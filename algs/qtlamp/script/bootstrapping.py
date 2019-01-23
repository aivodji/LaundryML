#!/usr/bin/env python
# coding:utf-8

import random, sys
from math import *
from subprocess import Popen, PIPE
from argparse import ArgumentParser

def bootstrapping(filename, outfilename, outLineNum):
  lines = open(filename, "r").readlines()
  if outfilename:
    out = open(outfilename, "w")
  else:
    out = sys.stdout
  for i in xrange(outLineNum):
    r = random.randint(0, len(lines)-1)
    out.write(lines[r])

def divide(filename, outfilename, divnum, size):
  print "disabled";sys.exit(0)
  lines = open(filename, "r").readlines()
  if outfilename:
    out_files = [open(outfilename+"."+str(i),"w") for i in range(1,divnum+1)]
    out_files_zero = open(outfilename+".0","w")
  else:
    print "error: outfilename is not specified"
    sys.exit(0)
  sizes = [0 for i in range(divnum)]
  for line in lines:
    r = random.randint(0, divnum-1)
    if sizes[r] < size:
      out_files[r].write(line)
      out_files_zero.write(line)
      sizes[r] += 1
  
def divide2(filename, outfilename, size):
  lines = open(filename, "r").readlines()
  divnum = 2
  if outfilename:
    out_files = [open(outfilename+"."+str(i),"w") for i in range(1,divnum+1)]
    out_files_zero = open(outfilename+".0","w")
  else:
    print "error: outfilename is not specified"
    sys.exit(0)
  sizes = [0 for i in range(divnum)]
  
  lines_files = [[] for i in range(divnum)]
  for line in lines:
    r = 0
    if random.random() < 0.2:
      r = 1
    if sizes[r] < size:
      lines_files[r].append(line)
      out_files_zero.write(line)
      sizes[r] += 1
  for i, line in enumerate(lines_files[0]):
    out_files[0].write(line)
    out_files[1].write(random.choice(lines_files[1]))
    
def getBootstrappedWinningRate(filename, patterns):
  patterns_s = map(set, patterns)
  lines = [line for line in open(filename, "r").readlines()]
  positives = [0 for p in patterns_s]
  nums = [0 for p in patterns_s]
  print len(lines),"lines"
  for line in lines:
    sps = map(int, line.strip().split())
    s = set(sps[:-1])
    label = sps[-1]
    for i,p in enumerate(patterns_s):
      #print p
      if p.issubset(s):
        nums[i] += 1
        if label > 0:
          positives[i] += 1
        else:
          pass
  ratios = [float(positives[i])/max(nums[i], 1) for i in range(len(patterns))]
  print "ratios=",ratios
  for i,pattern in enumerate(patterns):
    print " ".join(map(str, pattern)),ratios[i],[""," (FDR "+str(positives[i])+"/"+str(nums[i])+")"][ratios[i]<=0.3]
  return ratios
 
if __name__ == "__main__":
  parser = ArgumentParser()

  parser.add_argument(
      '-f', '--filename',
      type = str, 
      dest = 'filename', 
  )
  parser.add_argument(
      '-o', '--outfilename',
      type = str, 
      dest = 'outfilename', 
  )
  parser.add_argument(
      '-d', '--datafilename',
      type = str, 
      dest = 'datafilename', 
  )
  parser.add_argument(
      '-v', '--verif',
      action = 'store_true', 
      dest = 'verif'
  )
  parser.add_argument(
      '-s', '--split',
      action = 'store_true', 
      dest = 'split'
  )
  parser.add_argument(
      '-a', '--addtion',
      type = str,
      default = "",
      dest = 'addition',
  )
  parser.add_argument(
      '-n', '--numdata',
      type = int,
      default = 10000,
      dest = 'numdata',
  )
  parser.add_argument(
      '-w', '--wr',
      type = float,
      default = 0.3,
      dest = 'wr',
  )

  """ オプションをパース """
  args = parser.parse_args()
  verif, filename, outfilename, datafilename, addition, numdata = args.verif, args.filename, args.outfilename, args.datafilename, args.addition, args.numdata
  wr, split = args.wr, args.split
  
  if not filename:
    print "bootstrapping: file not specified";sys.exit(0)
    
  if split: #split data into calibration/main ones
    #print "split"
    divide2(filename, outfilename, numdata)
  elif verif:
    patterns = [map(int, line.split()) for line in open(filename,"r").readlines()]
    if addition == "":
      out = sys.stdout
    else:
      out = open(addition, "a")
    out.write( "Result of file "+filename+"\n" )
    out.write( "Num of patterns found: "+str(len(patterns))+"\n" )
    #out.write( "Num of false patterns found: "+str(len([ratio for ratio in ratios if ratio <= wr]))+"\n" )
    #out.write( "FDR :"+str(len([ratio for ratio in ratios if ratio <= wr])/float(len(ratios)))+"\n" )
    #if len([ratio for ratio in ratios if ratio <= wr]) >= 1:
    #  out.write( "False discovery found.\n" )
    #else:
    #  out.write( "False discovery NOT found.\n" )
  else: #resampling 
    bootstrapping(filename, outfilename, numdata)

 
