package edu.jhu.thrax.hadoop.features;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;

import edu.jhu.thrax.datatypes.AlignedSentencePair;
import edu.jhu.thrax.datatypes.Alignment;
import edu.jhu.thrax.hadoop.jobs.WordLexprobJob;
import edu.jhu.thrax.util.Vocabulary;
import edu.jhu.thrax.util.exceptions.MalformedInputException;
import edu.jhu.thrax.util.io.InputUtilities;

public class WordLexicalProbabilityCalculator extends Configured {
  public static final long UNALIGNED = 0x0000000000000000L;
  public static final long MARGINAL = 0x0000000000000000L;

  public static class Map extends Mapper<LongWritable, Text, LongWritable, IntWritable> {
    private HashMap<Long, Integer> counts = new HashMap<Long, Integer>();
    private boolean sourceParsed;
    private boolean targetParsed;
    private boolean reverse;
    private boolean sourceGivenTarget;

    protected void setup(Context context) throws IOException, InterruptedException {
      Configuration conf = context.getConfiguration();
      String vocabulary_path = conf.getRaw("thrax.work-dir") + "vocabulary/part-*";
      Vocabulary.initialize(conf, vocabulary_path);

      sourceParsed = conf.getBoolean("thrax.source-is-parsed", false);
      targetParsed = conf.getBoolean("thrax.target-is-parsed", false);
      reverse = conf.getBoolean("thrax.reverse", false);
      sourceGivenTarget = conf.getBoolean(WordLexprobJob.SOURCE_GIVEN_TARGET, false);
    }

    public void map(LongWritable key, Text value, Context context) throws IOException,
        InterruptedException {
      counts.clear();
      String line = value.toString();
      AlignedSentencePair sentencePair;
      try {
        sentencePair =
            InputUtilities.alignedSentencePair(line, sourceParsed, targetParsed,
                !(reverse ^ sourceGivenTarget));
      } catch (MalformedInputException e) {
        context.getCounter("input errors", e.getMessage()).increment(1);
        return;
      }
      int[] source = sentencePair.source;
      int[] target = sentencePair.target;
      Alignment alignment = sentencePair.alignment;

      for (int i = 0; i < source.length; i++) {
        int src = source[i];
        if (alignment.sourceIndexIsAligned(i)) {
          Iterator<Integer> target_indices = alignment.targetIndicesAlignedTo(i);
          while (target_indices.hasNext()) {
            int tgt = target[target_indices.next()];
            long pair = ((long) tgt << 32) + src;
            long marginal = ((long) tgt << 32) + MARGINAL;
            counts.put(pair, counts.containsKey(pair) ? counts.get(pair) + 1 : 1);
            counts.put(marginal, counts.containsKey(marginal) ? counts.get(marginal) + 1 : 1);
          }
        } else {
          long pair = UNALIGNED | ((long) src);
          long marginal = UNALIGNED;
          counts.put(pair, counts.containsKey(pair) ? counts.get(pair) + 1 : 1);
          counts.put(marginal, counts.containsKey(marginal) ? counts.get(marginal) + 1 : 1);
        }
      }
      for (long pair : counts.keySet())
        context.write(new LongWritable(pair), new IntWritable(counts.get(pair)));
    }
  }

  public static class Reduce
      extends Reducer<LongWritable, IntWritable, LongWritable, FloatWritable> {
    private int current = -1;
    private int marginalCount;

    protected void setup(Context context) throws IOException, InterruptedException {
      Configuration conf = context.getConfiguration();
      
      // TODO: remove unnecessary vocabulary loads?
      String vocabulary_path = conf.getRaw("thrax.work-dir") + "vocabulary/part-*";
      Vocabulary.initialize(conf, vocabulary_path);
    }

    protected void reduce(LongWritable key, Iterable<IntWritable> values, Context context)
        throws IOException, InterruptedException {
      long pair = key.get();
      int tgt = (int) (pair >> 32);
      int src = (int) (pair & 0xFFFFFFFFL);

      if (tgt != current) {
        if (src != MARGINAL) throw new RuntimeException("Sorting something before marginal.");
        current = tgt;
        marginalCount = 0;
        for (IntWritable x : values)
          marginalCount += x.get();
        return;
      }
      // Control only gets here if we are using the same marginal
      int my_count = 0;
      for (IntWritable x : values)
        my_count += x.get();
      context.write(key, new FloatWritable(my_count / (float) marginalCount));
    }
  }

  public static class Partition extends Partitioner<LongWritable, IntWritable> {
    public int getPartition(LongWritable key, IntWritable value, int numPartitions) {
      return ((int) (key.get() >> 32) & Integer.MAX_VALUE) % numPartitions;
    }
  }
}
