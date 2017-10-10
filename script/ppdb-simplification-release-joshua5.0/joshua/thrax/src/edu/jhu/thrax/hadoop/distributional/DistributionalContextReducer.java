package edu.jhu.thrax.hadoop.distributional;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import edu.jhu.jerboa.sim.SLSH;
import edu.jhu.jerboa.sim.Signature;

public class DistributionalContextReducer
    extends Reducer<Text, ContextWritable, SignatureWritable, NullWritable> {

  private int minCount;
  private SLSH slsh;

  public void setup(Context context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    minCount = conf.getInt("thrax.min-phrase-count", 3);
    slsh = CommonLSH.getSLSH(conf);
  }

  protected void reduce(Text key, Iterable<ContextWritable> values, Context context)
      throws IOException, InterruptedException {
    ContextWritable reduced = new ContextWritable();
    for (ContextWritable input : values) {
      reduced.merge(input, slsh);
    }
    if (!reduced.compacted.get()) reduced.compact(slsh);
    if (reduced.strength.get() >= minCount) {
      Signature reduced_signature = new Signature();
      // TODO: double-check need for deep copy?
      reduced_signature.sums = reduced.sums;
      slsh.buildSignature(reduced_signature, false);
      context.write(new SignatureWritable(key, reduced_signature, reduced.strength.get()),
          NullWritable.get());
    }
    return;
  }
}
