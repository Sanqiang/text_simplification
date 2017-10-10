package edu.jhu.thrax.hadoop.distributional;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import edu.jhu.jerboa.sim.SLSH;

public class DistributionalContextCombiner
    extends Reducer<Text, ContextWritable, Text, ContextWritable> {

  private SLSH slsh;

  public void setup(Context context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    slsh = CommonLSH.getSLSH(conf);
  }

  protected void reduce(Text key, Iterable<ContextWritable> values, Context context)
      throws IOException, InterruptedException {
    ContextWritable combined = new ContextWritable();
    for (ContextWritable input : values) {
      combined.merge(input, slsh);
    }
    if (!combined.compacted.get()) combined.compact(slsh);
    context.write(key, combined);
    return;
  }
}
