package edu.jhu.thrax.hadoop.distributional;

import org.apache.hadoop.conf.Configuration;

import edu.jhu.jerboa.sim.SLSH;

public class CommonLSH {

  public static SLSH getSLSH(Configuration conf) {
    SLSH slsh = null;
    try {
      slsh = new SLSH();
      slsh.initialize(conf.getInt("thrax.lsh-num-bits", 256),
          conf.getInt("thrax.lsh-pool-size", 100000), conf.getInt("thrax.lsh-random-seed", 42));
    } catch (Exception e) {
      e.printStackTrace();
      System.exit(1);
    }
    return slsh;
  }

}
