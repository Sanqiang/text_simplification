package parser;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import edu.stanford.nlp.ie.AbstractSequenceClassifier;
import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;

public class NERDataPrepare {
	AbstractSequenceClassifier<CoreLabel> classifier = null;
	HashSet<String> concat_tag_set = new HashSet<>(), replace_tag_set = new HashSet<>();

	public NERDataPrepare() throws Exception {
		classifier = CRFClassifier.getClassifier(
				"/Users/zhaosanqiang916/git/stanford-ner-2017-06-09/classifiers/english.conll.4class.distsim.crf.ser.gz");

		concat_tag_set.add("PERSON");
		concat_tag_set.add("LOCATION");
		concat_tag_set.add("ORGANIZATION");

		replace_tag_set.add("PERSON");
		replace_tag_set.add("LOCATION");
		replace_tag_set.add("ORGANIZATION");
		replace_tag_set.add("NUMBER");
	}

	public void process() throws Exception {
		String path_simp = "/Users/zhaosanqiang916/git/text_simplification_data/train/ours/simp350k.txt";
		String path_comp = "/Users/zhaosanqiang916/git/text_simplification_data/train/ours/comp350k.txt";
		String path_title = "/Users/zhaosanqiang916/git/text_simplification_data/train/ours/title350k.txt";
		BufferedReader reader_simp = new BufferedReader(new FileReader(new File(path_simp)));
		BufferedReader reader_comp = new BufferedReader(new FileReader(new File(path_comp)));
		BufferedReader reader_title = new BufferedReader(new FileReader(new File(path_title)));

		ArrayList<String> nlist_comp = new ArrayList<>(), nlist_simp = new ArrayList<>();

		String line_simp = null, line_comp = null, line_title = null;
		while (true) {
			line_comp = reader_comp.readLine();
			line_simp = reader_simp.readLine();
			line_title = reader_title.readLine();
			if (line_title == null) {
				if (line_comp != null || line_simp != null) {
					throw new Exception("Unequal lines for comp, simp, titles.");
				}
				break;
			}

			List<List<CoreLabel>> lcl_comp = classifier.classify(line_comp);
			List<List<CoreLabel>> lcl_simp = classifier.classify(line_simp);
			PaserResult paser_result = get_mapper(lcl_comp, lcl_simp);

			StringBuilder sb_comp = new StringBuilder(), sb_simp = new StringBuilder();

			for (String word : paser_result.list_comp) {
				if (paser_result.mapper.containsKey(word)) {
					sb_comp.append(paser_result.mapper.get(word));
				} else {
					sb_comp.append(word);
				}
				sb_comp.append(" ");
			}
			nlist_comp.add(sb_comp.toString());

			for (String word : paser_result.list_simp) {
				if (paser_result.mapper.containsKey(word)) {
					sb_simp.append(paser_result.mapper.get(word));
				} else {
					sb_simp.append(word);
				}
				sb_simp.append(" ");
			}
			nlist_simp.add(sb_simp.toString());
		}

		reader_comp.close();
		reader_simp.close();
		reader_title.close();
		
		String npath_simp = "/Users/zhaosanqiang916/git/text_simplification_data/train/ours/simp350.ner.txt";
		String npath_comp = "/Users/zhaosanqiang916/git/text_simplification_data/train/ours/comp350k.ner.txt";
		BufferedWriter writer_simp = new BufferedWriter(new FileWriter(new File(npath_simp)));
		BufferedWriter writer_comp = new BufferedWriter(new FileWriter(new File(npath_comp)));
		for (String line : nlist_simp) {
			writer_simp.write(line);
		}
		writer_simp.close();
		for (String line : nlist_comp) {
			writer_comp.write(line);
		}
		writer_comp.close();
	}

	private PaserResult get_mapper(List<List<CoreLabel>> lcl_comp, List<List<CoreLabel>> lcl_simp) {
		HashMap<String, String> mapper = new HashMap<>();
		String previous_tag = null;

		// The output ArrayList is for concat words
		ArrayList<String> list_simp = new ArrayList<>();
		ArrayList<String> list_comp = new ArrayList<>();

		for (List<CoreLabel> cls : lcl_simp) {
			for (CoreLabel coreLabel : cls) {
				String word = coreLabel.word();
				String ner = coreLabel.get(CoreAnnotations.AnswerAnnotation.class);
				if (!ner.equals("O")) {
					if (ner.equals(previous_tag)) {
						String concat_word = list_simp.get(list_simp.size() - 1) + " " + word;
						list_simp.set(list_simp.size() - 1, concat_word);
						mapper.remove(word);
						mapper.put(concat_word, ner);
					} else {
						list_simp.add(word);
						mapper.put(word, ner);
					}
					previous_tag = ner;
				} else {
					list_simp.add(word);
					previous_tag = null;
				}

				if (is_num(word)) {
					mapper.put(word, "NUMBER");
				}
			}
		}

		for (List<CoreLabel> cls : lcl_comp) {
			for (CoreLabel coreLabel : cls) {
				String word = coreLabel.word();
				String ner = coreLabel.get(CoreAnnotations.AnswerAnnotation.class);
				if (!ner.equals("O")) {
					if (ner.equals(previous_tag) && concat_tag_set.contains(ner)) {
						String concat_word = list_comp.get(list_comp.size() - 1) + " " + word;
						list_comp.set(list_comp.size() - 1, concat_word);
						mapper.remove(word);
						mapper.put(concat_word, ner);
					} else {
						list_comp.add(word);
						mapper.put(word, ner);
					}
					previous_tag = ner;
				} else {
					list_comp.add(replace_word(word));
					previous_tag = null;
				}

				if (is_num(word)) {
					mapper.put(word, "NUMBER");
				}
			}
		}

		HashMap<String, String> nmapper = new HashMap<>();
		HashMap<String, Integer> tag_cnt = new HashMap<>();
		for (String word : mapper.keySet()) {
			String tag = mapper.get(word);
			if (replace_tag_set.contains(tag)) {
				if (!tag_cnt.containsKey(tag)) {
					tag_cnt.put(tag, 1);
				}
				int id = tag_cnt.get(tag);
				nmapper.put(word, tag + "@" + id);
				tag_cnt.put(tag, id + 1);
			}
		}

		PaserResult result = new PaserResult();
		result.mapper = nmapper;
		result.list_comp = list_comp;
		result.list_simp = list_simp;
		return result;
	}

	class PaserResult {
		public HashMap<String, String> mapper;
		public ArrayList<String> list_simp, list_comp;

	}

	private String replace_word(String word) {
		if (word.equals("(")) {
			return "-LRB-";
		} else if (word.equals(")")) {
			return "-RRB-";
		} else {
			return word;
		}
	}

	private boolean is_num(String str) {
		try {
			Double.parseDouble(str);
		} catch (NumberFormatException nfe) {
			return false;
		}
		return true;
	}
	
	public static void main(String[] args) throws Exception {
		new NERDataPrepare().process();
	}
}
