package trans;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;

import org.apache.commons.lang3.Validate;
import org.apache.commons.text.StrBuilder;

import edu.stanford.nlp.ie.AbstractSequenceClassifier;
import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;

public class NERDataPrepare {
	AbstractSequenceClassifier<CoreLabel> classifier = null;
	HashSet<String> concat_tag_set = new HashSet<>(), replace_tag_set = new HashSet<>();

	public NERDataPrepare() throws Exception {
		classifier = CRFClassifier.getClassifier(
				"C:\\git\\java\\stanford-ner-2017-06-09\\classifiers\\english.conll.4class.distsim.crf.ser.gz");

		concat_tag_set.add("PERSON");
		concat_tag_set.add("LOCATION");
		concat_tag_set.add("ORGANIZATION");

		replace_tag_set.add("PERSON");
		replace_tag_set.add("LOCATION");
		replace_tag_set.add("ORGANIZATION");
		replace_tag_set.add("NUMBER");
	}

	public void process_multi(String path, String npath) throws Exception {
		File folder = new File(path);
		final File[] files = folder.listFiles();

		ForkJoinPool pool = new ForkJoinPool(7);
		pool.submit(() -> Arrays.asList(files).parallelStream().forEach(f -> {
			String name = f.getName();
			if (name.startsWith("simpf_")) {
				try {
					String id = name.substring(name.indexOf("_") + 1, name.indexOf(".txt"));
					File file_simp = f;
					File file_comp = new File(f.getParent() + "/compf_" + id + ".txt");
					process(file_simp, file_comp, npath, id);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		})).get();
	}

	public void process(File file_simp, File file_comp, String npath, String id) throws Exception {
		// String path_simp =
		// "/Users/zhaosanqiang916/git/text_simplification_data/train/ours/simp810.giga.txt";
		// String path_comp =
		// "/Users/zhaosanqiang916/git/text_simplification_data/train/ours/comp810.giga.txt";
		// String path_title =
		// "/Users/zhaosanqiang916/git/text_simplification_data/train/ours/title350k.txt";
		BufferedReader reader_simp = new BufferedReader(new FileReader(file_simp));
		BufferedReader reader_comp = new BufferedReader(new FileReader(file_comp));
		BufferedWriter writer_simp = new BufferedWriter(new FileWriter(new File(npath + "/ner_simp_" + id + ".txt")));
		BufferedWriter writer_comp = new BufferedWriter(new FileWriter(new File(npath + "/ner_comp_" + id + ".txt")));
		BufferedWriter writer_map = new BufferedWriter(new FileWriter(new File(npath + "/ner_map_" + id + ".txt")));
		// BufferedReader reader_title = new BufferedReader(new FileReader(new
		// File(path_title)));

		ArrayList<String> nlist_comp = new ArrayList<>(), nlist_simp = new ArrayList<>(), nlist_map = new ArrayList<>();
		int batch_id = 0;
		long cur_time = System.currentTimeMillis(), pre_time = cur_time;

		String line_simp = null, line_comp = null;
		while (true) {
			line_comp = reader_comp.readLine();
			line_simp = reader_simp.readLine();

			// line_title = reader_title.readLine();
			// if (line_title == null) {
			// if (line_comp != null || line_simp != null) {
			// throw new Exception("Unequal lines for comp, simp, titles.");
			// }
			// break;
			// }
			if (line_comp == null && line_simp == null) {
				break;
			}

			List<List<CoreLabel>> lcl_comp = classifier.classify(line_comp);
			List<List<CoreLabel>> lcl_simp = classifier.classify(line_simp);
			PaserResult paser_result = get_mapper(lcl_comp, lcl_simp);

			StringBuilder sb_comp = new StringBuilder(), sb_simp = new StringBuilder();

			for (String word : paser_result.list_comp) {
				if (paser_result.mapper.containsKey(word)) {
					String tag = paser_result.mapper.get(word);
					sb_comp.append(tag);
				} else {
					sb_comp.append(word);
				}
				sb_comp.append(" ");
			}
			nlist_comp.add(sb_comp.toString());

			for (String word : paser_result.list_simp) {
				if (paser_result.mapper.containsKey(word)) {
					String tag = paser_result.mapper.get(word);
					sb_simp.append(tag);
				} else {
					sb_simp.append(word);
				}
				sb_simp.append(" ");
			}
			nlist_simp.add(sb_simp.toString());

			StringBuilder sb_map = new StringBuilder();
			for (String word : paser_result.mapper.keySet()) {
				String tag = paser_result.mapper.get(word);
				sb_map.append(tag).append("=>").append(word).append("\t");
			}
			nlist_map.add(sb_map.toString());

			if (nlist_comp.size() % 10000 == 0) {
				for (String line : nlist_simp) {
					writer_simp.write(line);
					writer_simp.write("\n");
				}
				writer_simp.flush();
				for (String line : nlist_comp) {
					writer_comp.write(line);
					writer_comp.write("\n");
				}
				writer_comp.flush();
				for (String line : nlist_map) {
					writer_map.write(line);
					writer_map.write("\n");
				}
				writer_map.flush();
				nlist_simp.clear();
				nlist_comp.clear();
				nlist_map.clear();
				pre_time = cur_time;
				cur_time = System.currentTimeMillis();
				long span = (cur_time - pre_time) / 1000;

				System.out.println("Processed:\t" + ++batch_id + "\tfor id" + id + "\t using min\t" + span);
			}
		}

		reader_comp.close();
		reader_simp.close();
		// reader_title.close();

		for (String line : nlist_simp) {
			writer_simp.write(line);
			writer_simp.write("\n");
		}
		writer_simp.close();
		for (String line : nlist_comp) {
			writer_comp.write(line);
			writer_comp.write("\n");
		}
		writer_comp.close();
		for (String line : nlist_map) {
			writer_map.write(line);
			writer_map.write("\n");
		}
		writer_map.close();
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
						previous_tag = ner;
						String concat_word = list_simp.get(list_simp.size() - 1) + " " + word;
						list_simp.set(list_simp.size() - 1, concat_word);
						mapper.remove(word);
						mapper.put(concat_word, ner);
					} else {
						previous_tag = ner;
						list_simp.add(word);
						mapper.put(word, ner);
					}

				} else {
					list_simp.add(word);
					previous_tag = null;
				}

				if (is_num(word)) {
					mapper.put(word, "NUMBER");
				}
			}
		}

		previous_tag = null;
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

	public void postprocess(String path, String npath) throws Exception {
		File folder = new File(path);
		final File[] files = folder.listFiles();

		ForkJoinPool pool = new ForkJoinPool(7);
		pool.submit(() -> Arrays.asList(files).parallelStream().forEach(f -> {
			String name = f.getName();
			if (name.startsWith("ner_simp_")) {
				try {
					String id = name.substring(name.indexOf("simp_") + 5, name.indexOf(".txt"));
					File file_simp = f;
					File file_comp = new File(f.getParent() + "/ner_comp_" + id + ".txt");
					File file_map = new File(f.getParent() + "/ner_map_" + id + ".txt");

					BufferedReader reader_simp = new BufferedReader(new FileReader(file_simp));
					BufferedReader reader_comp = new BufferedReader(new FileReader(file_comp));
					BufferedReader reader_map = new BufferedReader(new FileReader(file_map));
					BufferedWriter writer_simp = new BufferedWriter(new FileWriter(npath + "/ner_simp_" + id + ".txt"));
					BufferedWriter writer_comp = new BufferedWriter(new FileWriter(npath + "/ner_comp_" + id + ".txt"));
					BufferedWriter writer_map = new BufferedWriter(new FileWriter(npath + "/ner_map_" + id + ".txt"));
					ArrayList<String> lines_simp = new ArrayList<>(), lines_comp = new ArrayList<>(),
							lines_map = new ArrayList<>();

					while (true) {
						String line_simp = reader_simp.readLine();
						String line_comp = reader_comp.readLine();
						String line_map = reader_map.readLine();
						if (line_comp == null || line_comp == null || line_map == null) {
							break;
						}

						String[] words_simp = line_simp.split(" ");
						String[] words_comp = line_comp.split(" ");
						HashSet<String> checker_simp = new HashSet<>(), checker_comp = new HashSet<>();
						for (String word : words_simp) {
							if (word.startsWith("PERSON@") || word.startsWith("LOCATION@")
									|| word.startsWith("ORGANIZATION@") || word.startsWith("NUMBER@")) {
								checker_simp.add(word);
							}
						}
						for (String word : words_comp) {
							if (word.startsWith("PERSON@") || word.startsWith("LOCATION@")
									|| word.startsWith("ORGANIZATION@") || word.startsWith("NUMBER@")) {
								checker_comp.add(word);
							}
						}

						boolean is_valid = true;
						for (String tag : checker_comp) {
							if (!checker_simp.contains(tag)) {
								is_valid = false;
							}
						}

						if (is_valid) {
							lines_comp.add(line_comp);
							lines_simp.add(line_simp);
							lines_map.add(line_map);
							if (lines_comp.size() >= 100000) {
								StringBuilder sb_comp = new StringBuilder();
								for (String line : lines_comp) {
									sb_comp.append(line).append("\n");
								}
								writer_comp.write(sb_comp.toString());
								writer_comp.flush();
								lines_comp.clear();

								StringBuilder sb_simp = new StringBuilder();
								for (String line : lines_simp) {
									sb_simp.append(line).append("\n");
								}
								writer_simp.write(sb_simp.toString());
								writer_simp.flush();
								lines_simp.clear();

								StringBuilder sb_map = new StringBuilder();
								for (String line : lines_map) {
									sb_map.append(line).append("\n");
								}
								writer_map.write(sb_map.toString());
								writer_map.flush();
								lines_map.clear();
							}
						}
					}
					StringBuilder sb_comp = new StringBuilder();
					for (String line : lines_comp) {
						sb_comp.append(line).append("\n");
					}
					writer_comp.write(sb_comp.toString());
					StringBuilder sb_simp = new StringBuilder();
					for (String line : lines_simp) {
						sb_simp.append(line).append("\n");
					}
					writer_simp.write(sb_simp.toString());
					StringBuilder sb_map = new StringBuilder();
					for (String line : lines_map) {
						sb_map.append(line).append("\n");
					}
					writer_map.write(sb_map.toString());

					reader_map.close();
					reader_simp.close();
					reader_comp.close();
					writer_map.close();
					writer_comp.close();
					writer_simp.close();

				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		})).get();
	}

	public void comebine2(String path, String npath, int batch_size, int file_size) throws Exception {
		ArrayList<ZipEntry> lines_zip = new ArrayList<>();
		for (int i = 0; i < 7; i++) {
			String path_simp = path + "/ner_comp_" + i + ".txt";
			String path_comp = path + "/ner_simp_" + i + ".txt";
			String path_map = path + "/ner_map_" + i + ".txt";
			BufferedReader reader_simp = new BufferedReader(new FileReader(new File(path_simp)));
			BufferedReader reader_comp = new BufferedReader(new FileReader(new File(path_comp)));
			BufferedReader reader_map = new BufferedReader(new FileReader(new File(path_map)));
			HashSet<String> set_comp = new HashSet<>(), set_simp = new HashSet<>();
			while (true) {
				String line_simp = reader_simp.readLine(), line_comp = reader_comp.readLine(),
						line_map = reader_map.readLine();
				if (line_comp == null || line_simp == null) {
					break;
				}
				if (!set_comp.contains(line_comp) && !set_simp.contains(line_simp)) {
					lines_zip.add(new ZipEntry(line_comp, line_simp, line_map));
					set_comp.add(line_comp);
					set_simp.add(line_simp);
				}
			}
			System.out.println("Finished file " + i);
			reader_comp.close();
			reader_simp.close();
			reader_map.close();
		}

		System.out.println("Populate with size " + lines_zip.size());
		Collections.shuffle(lines_zip);
		System.out.println("Shuffed the zip list!");

		int folder_id = 0, file_id = 0;
		int line_id = 0;
		StringBuilder sb_comp = new StringBuilder(), sb_simp = new StringBuilder(), sb_map = new StringBuilder();
		while (line_id < lines_zip.size()) {
			String folder_path = npath + "/" + folder_id + "/";
			File folder = new File(folder_path);
			if (!folder.exists()) {
				folder.mkdir();
				file_id = 0;
			}

			sb_comp.append(lines_zip.get(line_id).line_comp).append("\n");
			sb_simp.append(lines_zip.get(line_id).line_simp).append("\n");
			sb_map.append(lines_zip.get(line_id).line_map).append("\n");
			line_id++;

			if (line_id % batch_size == 0) {
				BufferedWriter writer_comp = new BufferedWriter(
						new FileWriter(new File(folder_path + "/comp" + file_id + ".txt")));
				BufferedWriter writer_simp = new BufferedWriter(
						new FileWriter(new File(folder_path + "/simp" + file_id + ".txt")));
				BufferedWriter writer_map = new BufferedWriter(
						new FileWriter(new File(folder_path + "/map" + file_id + ".txt")));
				writer_comp.write(sb_comp.toString());
				writer_comp.close();
				writer_simp.write(sb_simp.toString());
				writer_simp.close();
				writer_map.write(sb_map.toString());
				writer_map.close();
				file_id++;
				if (file_id % file_size == 0) {
					folder_id++;
				}
			}

		}
	}

	class ZipEntry {
		String line_comp, line_simp, line_map;

		public ZipEntry(String line_comp, String line_simp, String line_map) {
			this.line_comp = line_comp;
			this.line_simp = line_simp;
			this.line_map = line_map;
		}
	}

	public void combine(String path, String npath) throws Exception {
		BufferedWriter writer_comp = new BufferedWriter(new FileWriter(new File(npath + "/ner_comp.txt")));
		BufferedWriter writer_simp = new BufferedWriter(new FileWriter(new File(npath + "/ner_simp.txt")));
		BufferedWriter writer_map = new BufferedWriter(new FileWriter(new File(npath + "/ner_map.txt")));
		ArrayList<String> lines_comp = new ArrayList<>(), lines_simp = new ArrayList<>(), lines_map = new ArrayList<>();
		for (int i = 0; i < 7; i++) {
			String path_simp = path + "/ner_comp_" + i + ".txt";
			String path_comp = path + "/ner_simp_" + i + ".txt";
			String path_map = path + "/ner_map_" + i + ".txt";
			BufferedReader reader_simp = new BufferedReader(new FileReader(new File(path_simp)));
			BufferedReader reader_comp = new BufferedReader(new FileReader(new File(path_comp)));
			BufferedReader reader_map = new BufferedReader(new FileReader(new File(path_map)));
			while (true) {
				String line_simp = reader_simp.readLine(), line_comp = reader_comp.readLine(),
						line_map = reader_map.readLine();
				if (line_comp == null || line_simp == null) {
					break;
				}
				lines_comp.add(line_comp);
				lines_simp.add(line_simp);
				lines_map.add(line_map);
				if (lines_comp.size() >= 10000) {
					StringBuilder sb_comp = new StringBuilder();
					for (String line : lines_comp) {
						sb_comp.append(line).append("\n");
					}
					writer_comp.write(sb_comp.toString());
					writer_comp.flush();
					lines_comp.clear();

					StringBuilder sb_simp = new StringBuilder();
					for (String line : lines_simp) {
						sb_simp.append(line).append("\n");
					}
					writer_simp.write(sb_simp.toString());
					writer_simp.flush();
					lines_simp.clear();

					StringBuilder sb_map = new StringBuilder();
					for (String line : lines_map) {
						sb_map.append(line).append("\n");
					}
					writer_map.write(sb_map.toString());
					writer_map.flush();
					lines_map.clear();
				}
			}
			reader_comp.close();
			reader_simp.close();
			reader_map.close();
		}

		StringBuilder sb_comp = new StringBuilder();
		for (String line : lines_comp) {
			sb_comp.append(line).append("\n");
		}
		writer_comp.write(sb_comp.toString());
		writer_comp.close();

		StringBuilder sb_simp = new StringBuilder();
		for (String line : lines_simp) {
			sb_simp.append(line).append("\n");
		}
		writer_simp.write(sb_simp.toString());
		writer_simp.close();

		StringBuilder sb_map = new StringBuilder();
		for (String line : lines_map) {
			sb_map.append(line).append("\n");
		}
		writer_map.write(sb_map.toString());
		writer_map.close();
	}

	public static void main(String[] args) throws Exception {
		// new NERDataPrepare().process_multi("C:\\git\\wiki_output\\",
		// "C:\\git\\wiki_output\\ner");
		// new NERDataPrepare().postprocess("C:\\git\\wiki_output\\ner",
		// "C:\\git\\wiki_output\\ner2");
		new NERDataPrepare().comebine2("C:\\git\\wiki_output\\ner2", "C:\\git\\wiki_output\\ner4", 32, 500);
	}
}
