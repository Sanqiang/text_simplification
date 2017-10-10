package edu.jhu.thrax.hadoop.extraction;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import edu.jhu.thrax.datatypes.AlignedSentencePair;
import edu.jhu.thrax.datatypes.Alignment;
import edu.jhu.thrax.datatypes.HierarchicalRule;
import edu.jhu.thrax.datatypes.PhrasePair;
import edu.jhu.thrax.extraction.HierarchicalRuleExtractor;
import edu.jhu.thrax.extraction.HieroLabeler;
import edu.jhu.thrax.extraction.ManualSpanLabeler;
import edu.jhu.thrax.extraction.SAMTLabeler;
import edu.jhu.thrax.extraction.SpanLabeler;
import edu.jhu.thrax.hadoop.datatypes.AlignedRuleWritable;
import edu.jhu.thrax.hadoop.datatypes.AlignmentWritable;
import edu.jhu.thrax.hadoop.datatypes.Annotation;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.util.FormatUtils;
import edu.jhu.thrax.util.Vocabulary;
import edu.jhu.thrax.util.BackwardsCompatibility;
import edu.jhu.thrax.util.exceptions.MalformedInputException;
import edu.jhu.thrax.util.io.InputUtilities;

public class HierarchicalRuleWritableExtractor implements RuleWritableExtractor {
  private Mapper<LongWritable, Text, AlignedRuleWritable, Annotation>.Context context;

  private boolean sourceParsed;
  private boolean targetParsed;
  private boolean reverse;
  private boolean sourceLabels;
  private int defaultLabel;
  private int fullSentenceLabel;
  private int spanLimit;
  private AllowDefaultLabelPolicy allowDefaultLabel;

  private HierarchicalRuleExtractor extractor;

  public HierarchicalRuleWritableExtractor(
      Mapper<LongWritable, Text, AlignedRuleWritable, Annotation>.Context c) {
    context = c;
    Configuration conf = c.getConfiguration();
    sourceParsed = conf.getBoolean("thrax.source-is-parsed", false);
    targetParsed = conf.getBoolean("thrax.target-is-parsed", false);
    reverse = conf.getBoolean("thrax.reverse", false);
    // TODO: this configuration key needs a more general name now
    sourceLabels = !conf.getBoolean("thrax.target-is-samt-syntax", true);
    defaultLabel = Vocabulary.id(FormatUtils.markup(conf.get("thrax.default-nt", "X")));
    fullSentenceLabel = Vocabulary.id(FormatUtils.markup(conf.get("thrax.full-sentence-nt", "_S")));
    spanLimit = conf.getInt("thrax.initial-phrase-length", 10);
		setDefaultLabelPolicy(conf);
    extractor = getExtractor(conf);
  }

	private void setDefaultLabelPolicy(Configuration conf) {
		String policy;
		if (conf.get("thrax.allow-nonlexical-x") != null) {
			// allow-nonlexical-x was set to some value, so respect that
			policy = BackwardsCompatibility.defaultLabelPolicy(conf.getBoolean("thrax.allow-nonlexical-x", true));
		} else {
			policy = conf.get("thrax.allow-default-nt", "always");
		}
		allowDefaultLabel = AllowDefaultLabelPolicy.fromString(policy);
	}

  private static HierarchicalRuleExtractor getExtractor(Configuration conf) {
    int arity = conf.getInt("thrax.arity", 2);
    int initialPhraseSource = conf.getInt("thrax.initial-phrase-length", 10);
    int initialPhraseTarget = conf.getInt("thrax.initial-phrase-length", 10);
    int initialAlignment = conf.getInt("thrax.initial-lexicality", 1);
    boolean initialAligned = !conf.getBoolean("thrax.loose", false);
    int sourceLimit = conf.getInt("thrax.nonlex-source-length", 5);
    int targetLimit = conf.getInt("thrax.nonlex-target-length", 5);
    int lexSourceLimit = conf.getInt("thrax.lex-source-words", initialPhraseSource);
    int lexTargetLimit = conf.getInt("thrax.lex-target-words", initialPhraseTarget);
    int ruleAlignment = conf.getInt("thrax.lexicality", 1);
    boolean adjacent = conf.getBoolean("thrax.adjacent-nts", false);
    boolean abs = conf.getBoolean("thrax.allow-abstract-rules", false);
    boolean mixed = conf.getBoolean("thrax.allow-mixed-rules", true);
    boolean fullSentence = conf.getBoolean("thrax.allow-full-sentence-rules", true);
    return new HierarchicalRuleExtractor(arity, initialPhraseSource, initialPhraseTarget,
        initialAlignment, initialAligned, sourceLimit, targetLimit, ruleAlignment, adjacent, abs,
        mixed, fullSentence, lexSourceLimit, lexTargetLimit);
  }

  public Iterable<AnnotatedRule> extract(Text line) {
    AlignedSentencePair sentencePair;
    try {
      sentencePair =
          InputUtilities.alignedSentencePair(line.toString(), sourceParsed, targetParsed, reverse);
    } catch (MalformedInputException e) {
      context.getCounter("input errors", e.getMessage()).increment(1);
      return Collections.<AnnotatedRule>emptyList();
    }
    int[] source = sentencePair.source;
    int[] target = sentencePair.target;
    Alignment alignment = sentencePair.alignment;
    List<HierarchicalRule> rules = extractor.extract(source.length, target.length, alignment);
    List<AnnotatedRule> result = new ArrayList<AnnotatedRule>(rules.size());
    SpanLabeler labeler = getSpanLabeler(line, context.getConfiguration());
    if (labeler instanceof HieroLabeler) {
      // If this is false, we won't extract any rules, because the LHS label
      // is the default for Hiero.
      allowDefaultLabel = AllowDefaultLabelPolicy.ALWAYS;
    }
    for (HierarchicalRule r : rules) {
      RuleWritable rule = toRuleWritable(r, labeler, source, target);
      if (rule != null) {
        result.add(new AnnotatedRule(rule, new AlignmentWritable(r
            .compactSourceAlignment(alignment)), getRuleAnnotation(r, labeler, source, target,
            alignment)));
      }
    }
    return result;
  }

  private RuleWritable toRuleWritable(HierarchicalRule r, SpanLabeler spanLabeler, int[] source,
      int[] target) {
    int lhs;
    PhrasePair lhsPP = r.getLhs();
    int lhsSourceSpan = lhsPP.sourceEnd - lhsPP.sourceStart;
    int lhsTargetSpan = lhsPP.targetEnd - lhsPP.targetStart;
    if (lhsSourceSpan > spanLimit || lhsTargetSpan > spanLimit) {
      lhs = fullSentenceLabel;
    } else {
      lhs = r.lhsLabel(spanLabeler, sourceLabels);
    }
    int[] src = r.sourceSide(source, spanLabeler, sourceLabels);
    int[] tgt = r.targetSide(target, spanLabeler, sourceLabels);
    if (!isValidUseOfDefaultLabel(lhs, src, tgt)) return null;
    RuleWritable rw = new RuleWritable(lhs, src, tgt, r.monotonic());
    return rw;
  }

  private static Annotation getRuleAnnotation(HierarchicalRule r, SpanLabeler spanLabeler,
      int[] source, int[] target, Alignment alignment) {
    // TODO: this should be handling extraction-time annotation features.
    return new Annotation(1);
  }

  private SpanLabeler getSpanLabeler(Text line, Configuration conf) {
    String labelType = conf.get("thrax.grammar", "hiero");
    if (labelType.equalsIgnoreCase("hiero")) {
      return new HieroLabeler(defaultLabel);
    } else if (labelType.equalsIgnoreCase("samt")) {
      String[] fields = FormatUtils.P_DELIM.split(line.toString());
      if (fields.length < 2) return new HieroLabeler(defaultLabel);
      String parse = fields[sourceLabels ? 0 : 1].trim();
      boolean constituent = conf.getBoolean("thrax.allow-constituent-label", true);
      boolean ccg = conf.getBoolean("thrax.allow-ccg-label", true);
      boolean concat = conf.getBoolean("thrax.allow-concat-label", true);
      boolean double_concat = conf.getBoolean("thrax.allow-double-plus", true);
      String unary = conf.get("thrax.unary-category-handler", "all");
      return new SAMTLabeler(parse, constituent, ccg, concat, double_concat, unary, defaultLabel);
    } else if (labelType.equalsIgnoreCase("manual")) {
      String[] fields = FormatUtils.P_DELIM.split(line.toString());
      if (fields.length < 4) return new HieroLabeler(defaultLabel);
      int[] labels = Vocabulary.addAll(fields[3].trim());
      return new ManualSpanLabeler(labels, defaultLabel);
    } else {
      return new HieroLabeler(defaultLabel);
    }
  }

  private boolean isValidUseOfDefaultLabel(int lhs, int[] source, int[] target) {
		if (allowDefaultLabel == AllowDefaultLabelPolicy.ALWAYS) {
			return true;
		} else if (allowDefaultLabel == AllowDefaultLabelPolicy.PHRASES) {
      if (defaultLabel == lhs && hasNonterminal(source) && hasNonterminal(target))
        return false;
      else
        return !(hasNonterminal(source, defaultLabel) || hasNonterminal(target, defaultLabel));
    } else if (allowDefaultLabel == AllowDefaultLabelPolicy.NEVER) {
			return lhs != defaultLabel
				&& !hasNonterminal(source, defaultLabel)
				&& !hasNonterminal(target, defaultLabel);
		}
    return true;
  }

  private static boolean hasNonterminal(int[] s) {
    for (int w : s)
      if (w < 0) return true;
    return false;
  }

  private static boolean hasNonterminal(int[] s, int nt) {
    for (int w : s)
      if (w == nt) return true;
    return false;
  }

	private enum AllowDefaultLabelPolicy {
		ALWAYS, PHRASES, NEVER;

		public static AllowDefaultLabelPolicy fromString(String s) {
			if (s.equalsIgnoreCase("always")) {
				return ALWAYS;
			} else if (s.equalsIgnoreCase("phrases")) {
				return PHRASES;
			} else if (s.equalsIgnoreCase("never")) {
				return NEVER;
			} else {
				return ALWAYS;
			}
		}
	}
}
