package edu.jhu.thrax.hadoop.distributional;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import edu.jhu.thrax.distributional.ContextPhrase;
import edu.jhu.thrax.distributional.ContextPhraseExtractor;
import edu.jhu.thrax.util.MalformedInput;
import edu.jhu.thrax.util.exceptions.EmptySentenceException;
import edu.jhu.thrax.util.exceptions.MalformedInputException;
import edu.jhu.thrax.util.exceptions.MalformedParseException;
import edu.jhu.thrax.util.exceptions.NotEnoughFieldsException;

public class DistributionalContextMapper extends Mapper<LongWritable, Text, Text, ContextWritable> {

  private ContextPhraseExtractor extractor;

  protected void setup(Context context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    extractor = new ContextPhraseExtractor(conf);
  }

  protected void map(LongWritable key, Text value, Context context) throws IOException,
      InterruptedException {
    if (extractor == null) return;
    String line = value.toString();
    try {
      List<ContextPhrase> phrases = extractor.extract(line);
      for (ContextPhrase cp : phrases) {
        context.write(cp.getPhrase(), new ContextWritable(1, cp.getFeatures()));
      }
    } catch (NotEnoughFieldsException e) {
      context.getCounter(MalformedInput.NOT_ENOUGH_FIELDS).increment(1);
    } catch (EmptySentenceException e) {
      context.getCounter(MalformedInput.EMPTY_SENTENCE).increment(1);
    } catch (MalformedParseException e) {
      context.getCounter(MalformedInput.MALFORMED_PARSE).increment(1);
    } catch (MalformedInputException e) {
      context.getCounter(MalformedInput.UNKNOWN).increment(1);
    }
  }
}
