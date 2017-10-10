package joshua.metrics;

import java.util.logging.Logger;

public class FleschKincaid extends EvaluationMetric
{
  private static final Logger logger = Logger.getLogger(FleschKincaid.class.getName());

  //***(2)***
  private double maxGradeLevel = 25.0;
  private double minGradeLevel = 0.0;
  private int numsent = 1;
  //private ;

  /*
     You already have access to these data members of the parent
     class (EvaluationMetric):
         int numSentences;
           number of sentences in the MERT set
         int refsPerSen;
           number of references per sentence
         String[][] refSentences;
           refSentences[i][r] stores the r'th reference of the i'th
           source sentence (both indices are 0-based)
  */
 
  
  public FleschKincaid(String[] Metric_options)
  {

    //
    //
    // process the Metric_options array
    //
    //

    initialize(); // set the data members of the metric
  }

  protected void initialize()
  {
    //***(4)***
    metricName = "FK";    
    toBeMinimized = true; 
    suffStatsCount = 2;      
    		
    

    //***(5)***
    /* here you make calls to any methods that set the data members */
    /* here you make calls to any methods that set the data members */
    /* here you make calls to any methods that set the data members */
  }


  public double bestPossibleScore() { return minGradeLevel; }
  public double worstPossibleScore() { return maxGradeLevel; }


  /* here you define any methods that set the data members */
  /* here you define any methods that set the data members */
  /* here you define any methods that set the data members */

  
  public int[] suffStats(String cand_str, int i) 
  {
    int[] stats = new int[suffStatsCount];
    //stats[0]: number of tokens in the cand_str
    //stats[1]: number of syllables in the cand_str

    String[] candidate_tokens = null;

    if (!cand_str.equals("")) {
      candidate_tokens = cand_str.split("\\s+");
      if (candidate_tokens != null) {
    	  stats[0] = candidate_tokens.length;
    	  stats[1] = GradeLevelBLEU.countTotalSyllables(candidate_tokens);
      } else {
    	  System.out.println("DEBUG-FleschKincaid-SuffStats: " + cand_str + "|" + candidate_tokens.toString() + "!");
    	  stats[0] = 0;
    	  stats[1] = 0;
      }
      
    } else {
      candidate_tokens = new String[0];
      stats[0] = 0;
      stats[1] = 0;
    }

    return stats;
  }

  public double score(int[] stats)
  {
    if (stats.length != suffStatsCount) {
      System.out.println("Mismatch between stats.length and suffStatsCount (" + stats.length + " vs. " + suffStatsCount + ") in NewMetric.score(int[])");
      System.exit(1);
    }

    double sc = GradeLevelBLEU.gradeLevel(stats[0], stats[1], numsent);
    if (sc < minGradeLevel) {
    	sc = minGradeLevel;
    }
    
    if (sc > maxGradeLevel) {
    	sc = maxGradeLevel;
    }    

    //
    //
    // set sc here!
    //
    //

    return sc;
  }

  public void printDetailedScore_fromStats(int[] stats, boolean oneLiner)
  {
    System.out.println(metricName + " = " + score(stats));

    //
    //
    // optional (for debugging purposes)
    //
    //
  }

}

