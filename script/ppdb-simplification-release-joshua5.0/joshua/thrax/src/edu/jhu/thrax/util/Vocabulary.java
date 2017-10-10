package edu.jhu.thrax.util;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeMap;
import java.util.logging.Logger;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;

import edu.jhu.thrax.hadoop.features.SimpleFeature;
import edu.jhu.thrax.hadoop.features.SimpleFeatureFactory;
import edu.jhu.thrax.hadoop.features.annotation.AnnotationFeature;
import edu.jhu.thrax.hadoop.features.annotation.AnnotationFeatureFactory;
import edu.jhu.thrax.hadoop.features.mapred.MapReduceFeature;
import edu.jhu.thrax.hadoop.features.mapred.MapReduceFeatureFactory;
import edu.jhu.thrax.hadoop.features.pivot.PivotedAnnotationFeature;
import edu.jhu.thrax.hadoop.features.pivot.PivotedFeature;
import edu.jhu.thrax.hadoop.features.pivot.PivotedFeatureFactory;


/**
 * Static singular vocabulary class. Supports vocabulary freezing and (de-)serialization into a
 * vocabulary file.
 * 
 * @author Juri Ganitkevitch
 */

public class Vocabulary {

  private static final Logger logger;

  private static TreeMap<String, Integer> stringToId;
  private static ArrayList<String> idToString;

  private static int head;
  
  private static final Integer lock = new Integer(0);

  private static final int UNKNOWN_ID;
  private static final String UNKNOWN_WORD;

  public static final String START_SYM = "<s>";
  public static final String STOP_SYM = "</s>";


  static {
    logger = Logger.getLogger(Vocabulary.class.getName());

    UNKNOWN_ID = 0;
    UNKNOWN_WORD = "<unk>";

    clear();
  }

  /**
   * Reads a vocabulary from file. This deletes any additions to the vocabulary made prior to
   * reading the file.
   * 
   * @param file_name
   * @return Returns true if vocabulary was read without mismatches or collisions.
   * @throws IOException
   */
  public static boolean read(String file_name) throws IOException {
    synchronized (lock) {
      File vocab_file = new File(file_name);
      DataInputStream vocab_stream =
          new DataInputStream(new BufferedInputStream(new FileInputStream(vocab_file)));
      int size = vocab_stream.readInt();
      logger.info("Reading vocabulary: " + size + " tokens.");
      clear();
      for (int i = 0; i < size; i++) {
        int id = vocab_stream.readInt();
        String token = vocab_stream.readUTF();
        if (id != Math.abs(id(token))) {
          vocab_stream.close();
          return false;
        }
      }
      vocab_stream.close();
      return (size + 1 == idToString.size());
    }
  }

  /**
   * Initializes the vocabulary with the symbols defined by the config: feature labels and default
   * symbols. This deletes any additions to the vocabulary made prior to this call.
   */
  public static boolean initialize(Configuration conf) {
    synchronized (lock) {
      clear();
      // Add default symbols.
      id(FormatUtils.markup(conf.get("thrax.default-nt", "X")));
      id(FormatUtils.markup(conf.get("thrax.full-sentence-nt", "_S")));
      
      // Add feature names.
      String type = conf.get("thrax.type", "translation");
      String features = BackwardsCompatibility.equivalent(conf.get("thrax.features", ""));
      if ("translation".equals(type)) {
        for (MapReduceFeature f : MapReduceFeatureFactory.getAll(features))
          id(f.getLabel());
      } else if ("paraphrasing".equals(type)) {
        Set<String> prereq_features = new HashSet<String>();
        for (PivotedFeature f : PivotedFeatureFactory.getAll(features)) {
          prereq_features.addAll(f.getPrerequisites());
          id(f.getLabel());
        }
        id((new PivotedAnnotationFeature()).getLabel());
        for (String prereq : prereq_features) {
          MapReduceFeature mf = MapReduceFeatureFactory.get(prereq);
          if (mf != null) {
            id(mf.getLabel());
          } else {
            AnnotationFeature af = AnnotationFeatureFactory.get(prereq);
            if (af != null) {
              id(af.getLabel());
            } else {
              SimpleFeature sf = SimpleFeatureFactory.get(prereq);
              if (sf != null) id(sf.getLabel());
            }
          }
        }
      }
      for (AnnotationFeature f : AnnotationFeatureFactory.getAll(features))
        id(f.getLabel());
      for (SimpleFeature f : SimpleFeatureFactory.getAll(features))
        id(f.getLabel());
      head = size();      
      return true;
    }
  }

  /**
   * Initializes the vocabulary from a directory on HDFS. This deletes any additions to the
   * vocabulary made prior to reading the file.
   * 
   * @param conf
   * @param file
   * @return Returns true if vocabulary was read without mismatches or collisions.
   * @throws IOException
   */
  public static boolean initialize(Configuration conf, String file_glob) throws IOException {
    synchronized (lock) {
      FileSystem file_system = FileSystem.get(URI.create(file_glob), conf);
      FileStatus[] files = file_system.globStatus(new Path(file_glob));
      if (files.length == 0)
        throw new IOException("No files found in vocabulary glob: " + file_glob);

      initialize(conf);

      for (FileStatus file : files) {
        SequenceFile.Reader reader = new SequenceFile.Reader(file_system, file.getPath(), conf);
        Text h_token = new Text();
        IntWritable h_id = new IntWritable();
        while (reader.next(h_id, h_token)) {
          int id = h_id.get();
          String token = h_token.toString();
          if (!insert(token, id)) {
            reader.close();
            throw new RuntimeException("Error inserting: " + token + " as " + id + ". Conflict: "
                + id + " => " + Vocabulary.word(id) + " and " + token + " => "
                + stringToId.get(token));
          }
        }
        reader.close();
      }
      return true;
    }
  }

  public static void write(String file_name) throws IOException {
    synchronized (lock) {
      File vocab_file = new File(file_name);
      DataOutputStream vocab_stream =
          new DataOutputStream(new BufferedOutputStream(new FileOutputStream(vocab_file)));
      vocab_stream.writeInt(idToString.size() - 1);
      logger.info("Writing vocabulary: " + (idToString.size() - 1) + " tokens.");
      for (int i = 1; i < idToString.size(); i++) {
        vocab_stream.writeInt(i);
        vocab_stream.writeUTF(idToString.get(i));
      }
      vocab_stream.close();
    }
  }

  public static int id(String token) {
    synchronized (lock) {
      Integer id = stringToId.get(token);
      if (id != null) {
        return id;
      } else {
        int new_id = idToString.size() * (nt(token) ? -1 : 1);
        idToString.add(token);
        stringToId.put(token, new_id);
        return new_id;
      }
    }
  }

  private static boolean insert(String token, int set_id) {
    synchronized (lock) {
      Integer id = stringToId.get(token);
      if (id != null) {
        return (Math.abs(set_id) == Math.abs(id));
      } else {
        if (nt(token) && set_id > 0) set_id = -set_id;
        idToString.ensureCapacity(Math.abs(set_id) + 1);
        for (int i = idToString.size(); i <= Math.abs(set_id); ++i)
          idToString.add(null);
        idToString.set(Math.abs(set_id), token);
        stringToId.put(token, set_id);
        return true;
      }
    }
  }

  public static boolean hasId(int id) {
    synchronized (lock) {
      id = Math.abs(id);
      return (id < idToString.size());
    }
  }

  public static int[] addAll(String sentence) {
    synchronized (lock) {
      String[] tokens = FormatUtils.P_SPACE.split(sentence);
      int[] ids = new int[tokens.length];
      for (int i = 0; i < tokens.length; i++)
        ids[i] = id(tokens[i]);
      return ids;
    }
  }

  public static String word(int id) {
    synchronized (lock) {
      id = Math.abs(id);
      if (id >= idToString.size()) {
        throw new UnknownSymbolException(id);
      }
      return idToString.get(id);
    }
  }

  public static String getWords(int[] ids) {
    if (ids.length == 0) return "";
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < ids.length - 1; i++)
      sb.append(word(ids[i])).append(" ");
    return sb.append(word(ids[ids.length - 1])).toString();
  }

  public static String getWords(Iterable<Integer> ids) {
    StringBuilder sb = new StringBuilder();
    for (int id : ids)
      sb.append(word(id)).append(" ");
    return sb.deleteCharAt(sb.length() - 1).toString();
  }

  public static int getUnknownId() {
    return UNKNOWN_ID;
  }

  public static String getUnknownWord() {
    return UNKNOWN_WORD;
  }

  public static boolean nt(int id) {
    return (id < 0);
  }

  public static boolean idx(int id) {
    return (id < 0);
  }

  public static boolean nt(String word) {
    return FormatUtils.isNonterminal(word);
  }

  public static int size() {
    synchronized (lock) {
      return idToString.size();
    }
  }
  
  public static int head() {
    synchronized (lock) {
      return head;
    }
  }

  public static int getTargetNonterminalIndex(int id) {
    return FormatUtils.getNonterminalIndex(word(id));
  }

  private static void clear() {
    stringToId = new TreeMap<String, Integer>();
    idToString = new ArrayList<String>();

    idToString.add(UNKNOWN_ID, UNKNOWN_WORD);
  }

  /**
   * Used to indicate that a query has been made for a symbol that is not known.
   * 
   * @author Lane Schwartz
   */
  public static class UnknownSymbolException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    /**
     * Constructs an exception indicating that the specified identifier cannot be found in the
     * symbol table.
     * 
     * @param id Integer identifier
     */
    public UnknownSymbolException(int id) {
      super("Identifier " + id + " cannot be found in the symbol table");
    }

    /**
     * Constructs an exception indicating that the specified symbol cannot be found in the symbol
     * table.
     * 
     * @param symbol String symbol
     */
    public UnknownSymbolException(String symbol) {
      super("Symbol " + symbol + " cannot be found in the symbol table");
    }
  }

  /**
   * Used to indicate that word hashing has produced a collision.
   * 
   * @author Juri Ganitkevitch
   */
  public static class HashCollisionException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public HashCollisionException(String first, String second) {
      super("MurmurHash for the following symbols collides: '" + first + "', '" + second + "'");
    }
  }

  public static Iterator<String> wordIterator() {
    return idToString.iterator();
  }
}
