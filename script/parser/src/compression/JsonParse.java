package compression;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;

import org.json.JSONObject;

public class JsonParse {
	public void parse() throws Exception {
		ArrayList<String> list_normal = new ArrayList<>(), list_simple = new ArrayList<>();
		File folder = new File("/Volumes/Storage/sent_compression/");
		File[] files = folder.listFiles();
		int list_id = 0;
		for (File file : files) {
			BufferedReader reader = new BufferedReader(new FileReader(file));
			String line = null;
			StringBuilder sb_temp = new StringBuilder();
			while (null != (line = reader.readLine())) {
				sb_temp.append(line);
				if (line.equals("}")) {
					JSONObject jsonObj = new JSONObject(sb_temp.toString());
					sb_temp = new StringBuilder();
					String normal_sent = jsonObj.getJSONObject("graph").getString("sentence");
					String simple_sent = jsonObj.getJSONObject("compression").getString("text");
					list_normal.add(normal_sent);
					list_simple.add(simple_sent);

					if (list_normal.size() > 8000) {
						BufferedWriter writer_normal = new BufferedWriter(new FileWriter(
								new File("/Volumes/Storage/sent_compression_processed/normal" + list_id + ".txt")));
						BufferedWriter writer_simple = new BufferedWriter(new FileWriter(
								new File("/Volumes/Storage/sent_compression_processed/simple" + list_id + ".txt")));
						list_id += 1;
						System.out.println("Processed:" + list_id);
						for (int i = 0; i < list_normal.size(); i++) {
							writer_normal.write(list_normal.get(i));
							writer_normal.write("\n");
							writer_simple.write(list_simple.get(i));
							writer_simple.write("\n");
						}
						writer_normal.close();
						writer_simple.close();
						list_normal.clear();
						list_simple.clear();
					}
				}
			}
			reader.close();
			System.out.println("Finished current file:" + file.getName());
		}
		BufferedWriter writer_normal = new BufferedWriter(
				new FileWriter(new File("/Volumes/Storage/sent_compression_processed/normal" + list_id + ".txt")));
		BufferedWriter writer_simple = new BufferedWriter(
				new FileWriter(new File("/Volumes/Storage/sent_compression_processed/simple" + list_id + ".txt")));
		for (int i = 0; i < list_normal.size(); i++) {
			writer_normal.write(list_normal.get(i));
			writer_normal.write("\n");
			writer_simple.write(list_simple.get(i));
			writer_simple.write("\n");
		}
		writer_normal.flush();
		writer_simple.flush();
		writer_normal.close();
		writer_simple.close();
	}

	public static void main(String[] args) throws Exception {
		new JsonParse().parse();
	}
}
