package edu.jhu.thrax.extraction;

public class HieroLabeler implements SpanLabeler
{
    private final int label;

    public HieroLabeler(int s)
    {
        label = s;
    }

    public int getLabel(int start, int end)
    {
        return label;
    }
}

