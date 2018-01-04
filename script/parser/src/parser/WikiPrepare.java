package parser;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;

import com.google.common.base.CharMatcher;
import com.sun.org.apache.xpath.internal.operations.And;

import edu.stanford.nlp.ie.AbstractSequenceClassifier;
import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.DocumentPreprocessor;

public class WikiPrepare {
	AbstractSequenceClassifier<CoreLabel> classifier = null;
	HashMap<String, Double> tfidf = new HashMap<>();

	public WikiPrepare() throws Exception {
		classifier = CRFClassifier.getClassifier(
				"/Users/zhaosanqiang916/git/stanford-ner-2017-06-09/classifiers/english.conll.4class.distsim.crf.ser.gz");
	}

	private boolean is_num(String str) {
		try {
			Double.parseDouble(str);
		} catch (NumberFormatException nfe) {
			return false;
		}
		return true;
	}

	// For Pair Data
	public void extract_pair() throws Exception {
		init_tfidf();
		System.out.println("Init TFIDF.");
		// Extract title from simp files
		File folder = new File("/Volumes/Storage/wiki/text_simp/AA/");
		File[] files = folder.listFiles();
		// Loop Var
		String cur_title = null;
		StringBuilder sb_contnet = new StringBuilder();
		boolean start_doc = false;

		HashMap<String, ArrayList<String>> title2sents_simp = new HashMap<>();
		for (File file : files) {
			BufferedReader reader = new BufferedReader(new FileReader(file));
			String line = null;
			while (null != (line = reader.readLine())) {
				if (line.length() >= 4 && line.substring(0, 4).equals("<doc") && line.indexOf("title=") > 0
						&& line.indexOf("\">") > 0) {
					cur_title = line.substring(line.indexOf("title=") + 7, line.indexOf("\">"));
					title2sents_simp.put(cur_title, new ArrayList<>());
				} else if (line.length() >= 6 && line.substring(0, 6).equals("</doc>")) {
					// Process
					if (start_doc && cur_title != null) {
						DocumentPreprocessor dp = new DocumentPreprocessor(new StringReader(sb_contnet.toString()));
						for (List<HasWord> sentence : dp) {
							StringBuilder sb_tmp = new StringBuilder();
							for (HasWord hasWord : sentence) {
								sb_tmp.append(hasWord.word());
								sb_tmp.append(" ");
							}
							title2sents_simp.get(cur_title).add(sb_tmp.toString());
						}
					}
					start_doc = false;
					cur_title = null;
					sb_contnet = new StringBuilder();
				} else {
					if (start_doc && cur_title != null) {
						sb_contnet.append(line).append(" ");
					}
				}
			}
			reader.close();
		}

		// Extract comp files
		HashMap<String, ArrayList<String>> title2sents_comp = new HashMap<>();
		cur_title = null;
		sb_contnet = new StringBuilder();
		start_doc = false;

		// Outputs
		ArrayList<String> pair_comp = new ArrayList<>(), pair_simp = new ArrayList<>();
		BufferedWriter writer_comp = new BufferedWriter(
				new FileWriter(new File("/Volumes/Storage/wiki/comp_pair.txt"))),
				writer_simp = new BufferedWriter(new FileWriter(new File("/Volumes/Storage/wiki/simp_pair.txt")));

		File folder2 = new File("/Volumes/Storage/wiki/text_comp/AA/");
		File[] files2 = folder2.listFiles();
		for (File file : files2) {
			BufferedReader reader = new BufferedReader(new FileReader(file));
			String line = null;
			while (null != (line = reader.readLine())) {
				if (line.length() >= 4 && line.substring(0, 4).equals("<doc") && line.indexOf("title=") > 0
						&& line.indexOf("\">") > 0) {
					cur_title = line.substring(line.indexOf("title=") + 7, line.indexOf("\">"));
					if (title2sents_simp.containsKey(cur_title)) {
						start_doc = true;
					}
				} else if (line.length() >= 6 && line.substring(0, 6).equals("</doc>")) {
					// Process
					if (start_doc && cur_title != null) {
						ArrayList<String> sents_comp = new ArrayList<>();
						DocumentPreprocessor dp = new DocumentPreprocessor(new StringReader(sb_contnet.toString()));
						for (List<HasWord> sentence : dp) {
							StringBuilder sb_tmp = new StringBuilder();
							for (HasWord hasWord : sentence) {
								sb_tmp.append(hasWord.word());
								sb_tmp.append(" ");
							}
							sents_comp.add(sb_tmp.toString());
						}

						// Find the Simp sentences.
						for (String sent_simp : title2sents_simp.get(cur_title)) {
							String sent_comp = match(sent_simp, sents_comp);
							if (sent_comp != null) {
								pair_comp.add(sent_comp);
								pair_simp.add(sent_simp);

								if (pair_comp.size() >= 10000) {
									System.out.println("Batch Finished");
									assert pair_comp.size() == pair_simp.size();
									for (String tmp : pair_comp) {
										writer_comp.write(tmp);
										writer_comp.write("\n");
									}
									writer_comp.flush();
									pair_comp.clear();
									for (String tmp : pair_simp) {
										writer_simp.write(tmp);
										writer_simp.write("\n");
									}
									writer_simp.flush();
									pair_simp.clear();
									break;
								}
							}
						}

					}
					start_doc = false;
					cur_title = null;
					sb_contnet = new StringBuilder();
				} else {
					if (start_doc && cur_title != null) {
						sb_contnet.append(line);
					}
				}
			}
			reader.close();
		}
		assert pair_comp.size() == pair_simp.size();
		for (String tmp : pair_comp) {
			writer_comp.write(tmp);
			writer_comp.write("\n");
		}
		writer_comp.flush();
		writer_comp.close();
		for (String tmp : pair_simp) {
			writer_simp.write(tmp);
			writer_simp.write("\n");
		}
		writer_simp.flush();
		writer_simp.close();
	}

	private String match(String query, ArrayList<String> cands) {
		double max_sim = -1;
		String best_cand = null;
		for (String cand : cands) {
			double sim = get_sim(query, cand);
			if (sim > max_sim) {
				best_cand = cand;
				max_sim = sim;
			} else if (sim == max_sim) {
				return null;
			}
		}
		if (max_sim > 0 && best_cand != null) {
			cands.remove(best_cand);
			return best_cand;
		} else {
			return null;
		}

	}

	private double get_sim(String sent1, String sent2) {
		String[] words1 = sent1.split(" "), words2 = sent2.split(" ");
		ArrayList<String> list1 = new ArrayList<>(Arrays.asList(words1)),
				list2 = new ArrayList<>(Arrays.asList(words2));
		ArrayList<String> intersection_list = (ArrayList<String>) list1.clone();
		intersection_list.retainAll(list2);
		ArrayList<String> diff_list = (ArrayList<String>) list1.clone();
		diff_list.addAll(list2);
		diff_list.removeAll(intersection_list);

		double norm = 0.0, denorm = 0.0;
		for (String word : intersection_list) {
			norm += tfidf.get(word);
			denorm += tfidf.get(word);
		}
		for (String word : diff_list) {
			denorm += tfidf.get(word);
		}
		return norm / denorm;
	}

	private void init_tfidf() throws Exception {
		HashMap<String, Integer> word2cnt = new HashMap<>(), title2idx = new HashMap<>();
		HashMap<String, HashSet<Integer>> word2titles = new HashMap<>();
		double title_cnt = 0;

		String[] paths = { "/Volumes/Storage/wiki/simp_all_title.txt", "/Volumes/Storage/wiki/comp_all_title.txt" };
		for (String path : paths) {
			BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
			String line = null;
			String cur_title = null;
			while (null != (line = reader.readLine())) {
				if (line.length() > 5 && line.substring(0, 5).equals("=====")) {
					cur_title = line.substring(5, line.lastIndexOf("====="));
					if (!title2idx.containsKey(cur_title)) {
						title2idx.put(cur_title, (int) title_cnt);
						title_cnt += 1;
					}
				} else {
					String[] words = line.split(" ");
					for (String word : words) {
						word = word.toLowerCase();
						if (!word2cnt.containsKey(word)) {
							word2cnt.put(word, 0);
						}
						word2cnt.put(word, 1 + word2cnt.get(word));

						if (!word2titles.containsKey(word)) {
							word2titles.put(word, new HashSet<>());
						}
						word2titles.get(word).add(title2idx.get(cur_title));
					}
				}
				System.out.println("Word Cnt:" + word2cnt.size());
			}
			reader.close();
		}

		for (String word : word2cnt.keySet()) {
			tfidf.put(word, Math.log(1.0 + word2cnt.get(word) * Math.log(title_cnt / word2titles.get(word).size())));
		}
	}

	// For LM model use
	public void extract_wiki_lm() throws Exception {
		boolean use_ner = false, use_title = false, sep_files = true;
		File folder = new File("/Volumes/Storage/wiki/text_comp/AA/");
		File[] files = folder.listFiles();
		// Loop Var
		long line_id = 0;
		String cur_title = null;
		StringBuilder sb_contnet = new StringBuilder();
		boolean start_doc = false;
		// Output Var
		ArrayList<String> output_sents = new ArrayList<>();
		BufferedWriter writer = new BufferedWriter(new FileWriter(new File("/Volumes/Storage/wiki/comp_all.txt")));
		for (File file : files) {
			BufferedReader reader = new BufferedReader(new FileReader(file));
			String line = null;
			while (null != (line = reader.readLine())) {
				if (line.length() >= 4 && line.substring(0, 4).equals("<doc") && line.indexOf("title=") > 0
						&& line.indexOf("\">") > 0) {
					cur_title = line.substring(line.indexOf("title=") + 7, line.indexOf("\">"));
					start_doc = true;
				} else if (line.length() >= 6 && line.substring(0, 6).equals("</doc>")) {
					// Process
					DocumentPreprocessor dp = new DocumentPreprocessor(new StringReader(sb_contnet.toString()));
					if (use_title) {
						output_sents.add("=====" + cur_title + "=====");
					}

					if (use_ner) {
						int sent_cnt = 0;
						StringBuilder sb_tmp_doc = new StringBuilder();
						for (List<HasWord> sentence : dp) {
							for (HasWord hasWord : sentence) {
								sb_tmp_doc.append(hasWord.word());
								sb_tmp_doc.append(" ");
							}
							++sent_cnt;
						}
						List<List<CoreLabel>> sents = classifier.classify(sb_tmp_doc.toString());
						assert sents.size() == sent_cnt;
						for (List<CoreLabel> sent : sents) {
							StringBuilder sb_tmp = new StringBuilder();
							String pre_ner = "";
							for (CoreLabel coreLabel : sent) {
								String ner = coreLabel.get(CoreAnnotations.AnswerAnnotation.class);
								if (is_num(coreLabel.word())) {
									ner = "NUMBER";
								}
								if (ner.equals("O")) {
									sb_tmp.append(coreLabel.word());
									sb_tmp.append(" ");
									pre_ner = "";
								} else {
									if (!pre_ner.equals("") && pre_ner.equals(ner)) {
									} else {
										sb_tmp.append(ner);
										sb_tmp.append(" ");
									}
									pre_ner = ner;
								}
							}
							output_sents.add(sb_tmp.toString());
						}
					} else {
						boolean add_sent = true;
						StringBuilder sb_tmp = new StringBuilder();
						for (List<HasWord> sentence : dp) {
							for (HasWord hasWord : sentence) {
								if (!CharMatcher.ascii().matchesAllOf(hasWord.word())) {
									add_sent = false;
									break;
								}
								sb_tmp.append(hasWord.word());
								sb_tmp.append(" ");
							}
							String last_word = sentence.get(sentence.size() - 1).word();
							if (last_word.equals(".") || last_word.equals("?") || last_word.equals("!")) {
								if (sb_tmp.toString().split(" ").length >= 10 && add_sent) {
									output_sents.add(sb_tmp.toString());
									add_sent = true;
								}
								sb_tmp = new StringBuilder();
							}
						}
					}

					if (output_sents.size() >= 5000) {
						if (sep_files) {
							writer = new BufferedWriter(new FileWriter(
									new File("/Volumes/Storage/wiki/wiki_output/comp/" + line_id + ".txt")));
						}
						line_id += 1;
						System.out.println("Processed " + line_id);
						for (String sent : output_sents) {
							writer.write(sent);
							writer.write("\n");
						}
						writer.flush();
						output_sents.clear();
						if (sep_files) {
							writer.close();
						}
					}

					start_doc = false;
					cur_title = null;
					sb_contnet = new StringBuilder();
				} else {
					if (start_doc && cur_title != null) {
						sb_contnet.append(line).append(" ");
					}
				}
			}
			reader.close();
		}
		if (sep_files) {
			writer = new BufferedWriter(
					new FileWriter(new File("/Volumes/Storage/wiki/wiki_output/comp/" + line_id + ".txt")));
		}
		for (String sent : output_sents) {
			writer.write(sent);
			writer.write("\n");
		}
		writer.flush();
		writer.close();

	}

	// public void extract_wiki_lm_ner() throws Exception {
	// BufferedWriter writer = new BufferedWriter(
	// new FileWriter(new File("/Volumes/Storage/wiki/simp_all_title_ner.txt")));
	//
	// List<List<CoreLabel>> sents =
	// classifier.classifyFile("/Volumes/Storage/wiki/simp_all_title.txt");
	// StringBuilder sBuilder = new StringBuilder();
	// for (List<CoreLabel> sent : sents) {
	// for (CoreLabel coreLabel : sent) {
	// String ner = coreLabel.get(CoreAnnotations.AnswerAnnotation.class);
	// if (ner.equals("O")) {
	// sBuilder.append(coreLabel.word());
	// } else {
	// sBuilder.append(ner);
	// }
	// sBuilder.append(" ");
	// }
	// sBuilder.append("\n");
	// if (sBuilder.length() > 10000) {
	// writer.write(sBuilder.toString());
	// writer.flush();
	// sBuilder = new StringBuilder();
	// }
	// }
	// writer.write(sBuilder.toString());
	// writer.flush();
	// writer.close();
	// }

	public static void main(String[] args) throws Exception {
		new WikiPrepare().extract_wiki_lm();
		// new WikiPrepare().extract_wiki_lm_ner();
		// new WikiPrepare().extract_pair();
	}
}
