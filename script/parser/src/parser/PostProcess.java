package parser;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class PostProcess {

	private final int THRESHOLD = 3000;

	private HashMap<String, Integer> tf = new HashMap<>();
	private String path = null, npath = null;
	private HashSet<String> tags = new HashSet<>();

	public PostProcess(String path, String npath) {
		this.path = path;
		this.npath = npath;

		this.tags.add("LOCATION");
		this.tags.add("ORGANIZATION");
		this.tags.add("MISC");
		this.tags.add("PERSON");
	}

	private void populate_tf() throws Exception {
		BufferedReader reader = new BufferedReader(new FileReader(new File(this.path)));
		String line = null;
		while (null != (line = reader.readLine())) {
			if (line.startsWith("=====")) {
				continue;
			}
			String[] words = line.split(" ");
			for (String word : words) {
				if (!this.tf.containsKey(word)) {
					this.tf.put(word, 0);
				}
				this.tf.put(word, this.tf.get(word) + 1);
			}
		}
		reader.close();

		String[] files = { "/Users/zhaosanqiang916/git/text_simplification_data/val/wiki.full.aner.ori.valid.src",
				"/Users/zhaosanqiang916/git/text_simplification_data/val/wiki.full.aner.ori.valid.dst",
				"/Users/zhaosanqiang916/git/text_simplification_data/test/wiki.full.aner.test.src",
				"/Users/zhaosanqiang916/git/text_simplification_data/test/wiki.full.aner.test.dst" };
		for (String file : files) {
			reader = new BufferedReader(new FileReader(new File(file)));
			line = null;
			while (null != (line = reader.readLine())) {
				String[] words = line.split(" ");
				for (String word : words) {
					word = word.replace("@", "");
					if (!this.tf.containsKey(word)) {
						this.tf.put(word, THRESHOLD);
					}
					this.tf.put(word, this.tf.get(word) + 1);
				}
			}
			reader.close();
		}

		System.out.println("Populate TF!");
	}

	public void process() throws Exception {
		this.populate_tf();
		BufferedReader reader = new BufferedReader(new FileReader(new File(this.path)));
		BufferedWriter writer = new BufferedWriter(new FileWriter(new File(this.npath)));
		String line = null;
		long line_id = 0;
		long unk_id = 0;
		ArrayList<String> nlines = new ArrayList<>();
		while (null != (line = reader.readLine())) {
			if (line.startsWith("=====")) {
				continue;
			}
			++line_id;
			int tag_per = 1, tag_loc = 1, tag_org = 1, tag_misc = 1;
			StringBuilder sb_nline = new StringBuilder();
			String[] words = line.split(" ");
			boolean is_valid = true;
			for (int i = 0; i < words.length; i++) {
				String word = words[i];
				if (this.tf.get(word) < THRESHOLD) {
					is_valid = false;
					unk_id++;
					break;
				}
				if (i > 0 && tags.contains(word) && word.equals(words[i - 1])) {

				} else {
					if (tags.contains(word)) {
						if (word.equals("PERSON")) {
							sb_nline.append(word + tag_per);
							tag_per++;
						} else if (word.equals("LOCATION")) {
							sb_nline.append(word + tag_loc);
							tag_loc++;
						} else if (word.equals("ORGANIZATION")) {
							sb_nline.append(word + tag_org);
							tag_org++;
						} else if (word.equals("MISC")) {
							sb_nline.append(word + tag_misc);
							tag_misc++;
						}
					} else {
						sb_nline.append(word);
					}
					sb_nline.append(" ");
				}
			}
			if (is_valid) {
				nlines.add(sb_nline.toString());
				if (nlines.size() >= 1000000) {
					for (String nline : nlines) {
						writer.write(nline);
						writer.write("\n");
					}
					writer.flush();
					nlines.clear();
					System.out.println("line id:" + line_id);
				}
			}
		}
		reader.close();

		for (String nline : nlines) {
			writer.write(nline);
			writer.write("\n");
		}
		writer.close();
		System.out.println("UNK CNT:" + unk_id);
	}

	public static void main(String[] args) throws Exception {
		PostProcess postProcess = new PostProcess("/Volumes/Storage/wiki/comp_all_refined_trans_thres3000_19.txt",
				"/Volumes/Storage/wiki/comp_all_refined_trans_thres3000_20.txt");
		postProcess.process();
	}

}
