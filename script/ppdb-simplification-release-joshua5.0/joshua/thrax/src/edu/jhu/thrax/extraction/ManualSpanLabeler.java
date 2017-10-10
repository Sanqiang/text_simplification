package edu.jhu.thrax.extraction;

public class ManualSpanLabeler implements SpanLabeler
{
    private final int [] labels;
	private final int defaultLabel;
    private final int sentenceLength;

    public ManualSpanLabeler(int[] ls, int def)
    {
		labels = ls;
		defaultLabel = def;
		sentenceLength = getSentenceLength(labels.length);
    }

    public int getLabel(int from, int to)
    {
        int idx = getLabelIndex(from, to, sentenceLength);
        if (idx >= labels.length || idx < 0) {
            return defaultLabel;
        }
        else {
            return labels[idx];
        }
    }

    private static int getSentenceLength(int numLabels)
    {
        if (numLabels < 0)
            return 0;
        // 0 labels => sentence length 0
        // 1 label => 1
        // 3 labels => 2
        // T_n labels => n, where T_n is the nth triangle number
        int result = 0;
        int triangle = 0;
        while (triangle != numLabels) {
            result++;
            triangle += result;
        }
        return result;
    }

    private static int getLabelIndex(int from, int to, int length)
    {
        // let the length of the target sentence be L
        // the first L labels are for spans (0,1) ... (0,L)
        // the next L - 1 are for (1,2) ... (1,L)
        // and so on
        int result = 0;
        int offset = length;
        for (int i = 0; i < from; i++) {
            result += offset;
            offset--;
        }
        int difference = to - from - 1;
        result += difference;
        return result;
    }
}

