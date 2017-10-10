package edu.jhu.thrax.hadoop.jobs;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import edu.jhu.thrax.extraction.Labeling;
import edu.jhu.thrax.util.FormatUtils;
import edu.jhu.thrax.util.Vocabulary;

public class VocabularyJob implements ThraxJob {

  public VocabularyJob() {}

  public Job getJob(Configuration conf) throws IOException {
    Job job = new Job(conf, "vocabulary");
    job.setJarByClass(VocabularyJob.class);

    job.setMapperClass(VocabularyJob.Map.class);
    job.setCombinerClass(VocabularyJob.Combine.class);
    job.setReducerClass(VocabularyJob.Reduce.class);

    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(NullWritable.class);

    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(Text.class);

    job.setSortComparatorClass(Text.Comparator.class);

    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    FileInputFormat.setInputPaths(job, new Path(conf.get("thrax.input-file")));
    int maxSplitSize = conf.getInt("thrax.max-split-size", 0);
    if (maxSplitSize != 0) FileInputFormat.setMaxInputSplitSize(job, maxSplitSize);

    FileOutputFormat.setOutputPath(job, new Path(conf.get("thrax.work-dir") + "vocabulary"));

    int num_reducers = conf.getInt("thrax.reducers", 4);
    job.setNumReduceTasks(num_reducers);

    return job;
  }

  public String getOutputSuffix() {
    return "vocabulary";
  }

  @Override
  public String getName() {
    return "vocabulary";
  }

  private static class Map extends Mapper<LongWritable, Text, Text, NullWritable> {

    private boolean sourceParsed;
    private boolean targetParsed;
    private Labeling labeling;

    private boolean allowConstituent = true;
    private boolean allowCCG = true;
    private boolean allowConcat = true;
    private boolean allowDoubleConcat = true;

    protected void setup(Context context) {
      Configuration conf = context.getConfiguration();
      sourceParsed = conf.getBoolean("thrax.source-is-parsed", false);
      targetParsed = conf.getBoolean("thrax.target-is-parsed", false);

      allowConstituent = conf.getBoolean("thrax.allow-constituent-label", true);
      allowCCG = conf.getBoolean("thrax.allow-ccg-label", true);
      allowConcat = conf.getBoolean("thrax.allow-concat-label", true);
      allowDoubleConcat = conf.getBoolean("thrax.allow-double-plus", true);

      if (conf.get("thrax.grammar", "hiero").equalsIgnoreCase("samt")) {
        labeling = Labeling.SYNTAX;
      } else if (conf.get("thrax.grammar", "hiero").equalsIgnoreCase("manual")) {
        labeling = Labeling.MANUAL;
      } else {
        labeling = Labeling.HIERO;
      }
    }

    protected void map(LongWritable key, Text input, Context context) throws IOException,
        InterruptedException {

      String[] parts = FormatUtils.P_DELIM.split(input.toString());
      if (parts.length < 3) return;

      if (sourceParsed)
        extractTokensFromParsed(parts[0], (labeling != Labeling.SYNTAX), context);
      else
        extractTokens(parts[0], context);

      if (targetParsed)
        extractTokensFromParsed(parts[1], (labeling != Labeling.SYNTAX), context);
      else
        extractTokens(parts[1], context);

      if (labeling == Labeling.MANUAL && parts.length > 3) {
        String[] labels = FormatUtils.P_SPACE.split(parts[3].trim());
        for (String label : labels)
          context.write(new Text("[" + label), NullWritable.get());
      }
    }

    protected void extractTokens(String input, Context context) throws IOException,
        InterruptedException {
      if (input == null || input.isEmpty()) return;
      String[] tokens = FormatUtils.P_SPACE.split(input);
      for (String token : tokens)
        if (!token.isEmpty()) context.write(new Text(token), NullWritable.get());
    }

    protected void extractTokensFromParsed(String input, boolean terminals_only, Context context)
        throws IOException, InterruptedException {
      int from = 0, to = 0;
      boolean seeking = true;
      boolean nonterminal = false;
      char current;

      Set<String> nonterminals = new HashSet<String>();

      if (input == null || input.isEmpty() || input.charAt(0) != '(') return;

      // Run through entire (potentially parsed) sentence.
      while (from < input.length() && to < input.length()) {
        if (seeking) {
          // Seeking mode: looking for the start of the next symbol.
          current = input.charAt(from);
          if (current == '(' || current == ')' || current == ' ') {
            // We skip brackets and spaces.
            ++from;
          } else {
            // Found a non spacing symbol, go into word filling mode.
            to = from + 1;
            seeking = false;
            nonterminal = (input.charAt(from - 1) == '(');
          }
        } else {
          // Word filling mode. Advance to until we hit the end or spacing.
          current = input.charAt(to);
          if (current == ' ' || current == ')' || current == '(') {
            // Word ended.
            if (terminals_only) {
              if (!nonterminal)
                context.write(new Text(input.substring(from, to)), NullWritable.get());
            } else {
              if (nonterminal) {
                String nt = input.substring(from, to);
                if (nt.equals(",")) nt = "COMMA";
                nonterminals.add("[" + nt);
              } else {
                context.write(new Text(input.substring(from, to)), NullWritable.get());
              }
            }
            from = to + 1;
            seeking = true;
          } else {
            ++to;
          }
        }
      }
      if (!terminals_only) combineNonterminals(context, nonterminals);
    }


    private void combineNonterminals(Context context, Set<String> nonterminals) throws IOException,
        InterruptedException {
      if (allowConstituent) writeNonterminals(nonterminals, context);
      if (allowConcat) {
        Set<String> concatenated = joinNonterminals("+", nonterminals, nonterminals);
        writeNonterminals(concatenated, context);
      }
      if (allowCCG) {
        Set<String> forward = joinNonterminals("/", nonterminals, nonterminals);
        writeNonterminals(forward, context);
        Set<String> backward = joinNonterminals("\\", nonterminals, nonterminals);
        writeNonterminals(backward, context);
      }
      if (allowDoubleConcat) {
        Set<String> concat = joinNonterminals("+", nonterminals, nonterminals);
        Set<String> double_concat = joinNonterminals("+", concat, nonterminals);
        writeNonterminals(double_concat, context);
      }
    }

    private Set<String> joinNonterminals(String glue, Set<String> prefixes, Set<String> nonterminals) {
      Set<String> joined = new HashSet<String>();
      for (String prefix : prefixes)
        for (String nt : nonterminals)
          joined.add(prefix + glue + nt.substring(1));
      return joined;
    }

    private static void writeNonterminals(Set<String> nts, Context context) throws IOException,
        InterruptedException {
      for (String nt : nts)
        context.write(new Text(nt + "]"), NullWritable.get());
    }

  }

  public static class VocabularyPartitioner extends Partitioner<Text, Writable> {
    public int getPartition(Text key, Writable value, int numPartitions) {
      return (key.hashCode() & Integer.MAX_VALUE) % numPartitions;
    }
  }

  private static class Combine extends Reducer<Text, NullWritable, Text, NullWritable> {

    protected void reduce(Text key, Iterable<NullWritable> values, Context context)
        throws IOException, InterruptedException {
      context.write(key, NullWritable.get());
    }
  }

  private static class Reduce extends Reducer<Text, NullWritable, IntWritable, Text> {

    private int reducerNumber;
    private int numReducers;

    protected void setup(Context context) throws IOException, InterruptedException {
      numReducers = context.getNumReduceTasks();
      reducerNumber = context.getTaskAttemptID().getTaskID().getId();
      
      Vocabulary.initialize(context.getConfiguration());
    }

    protected void reduce(Text key, Iterable<NullWritable> values, Context context)
        throws IOException, InterruptedException {
      String token = key.toString();
      if (token == null || token.isEmpty()) throw new RuntimeException("Unexpected empty token.");
      Vocabulary.id(token);
      context.progress();
    }

    protected void cleanup(Context context) throws IOException, InterruptedException {
      for (int i = Vocabulary.head(); i < Vocabulary.size(); ++i)
        context.write(new IntWritable((i - 1) * numReducers + reducerNumber + 1), new Text(
            Vocabulary.word(i)));
    }
  }

  @Override
  public Set<Class<? extends ThraxJob>> getPrerequisites() {
    return new HashSet<Class<? extends ThraxJob>>();
  }
}
