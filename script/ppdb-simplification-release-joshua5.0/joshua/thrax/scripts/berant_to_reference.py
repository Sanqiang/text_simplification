#!/usr/bin/env python

import os, sys, codecs

def main():
  # <eat::animal_1::animal_2>	<kill::animal_1::animal_2>
  # <be affected than::animal_2::animal_1>	<be susceptible than::animal_2::animal_1>
  for line in sys.stdin:
    (source, target) = line.lstrip().rstrip().split("\t")
    (s_phr, s1, s2) = source[1:-1].split("::")
    (t_phr, t1, t2) = target[1:-1].split("::")
    if (s1[-2:] == t1[-2:]):
      t1 = "[1]"
      t2 = "[2]"
    else:
      t1 = "[2]"
      t2 = "[1]"
    s1 = "[1]"
    s2 = "[2]"
    print s1 + " " + s_phr + " " + s2 + " ||| " + t1 + " " + t_phr + " " + t2
    

if __name__ == "__main__":
    main()


