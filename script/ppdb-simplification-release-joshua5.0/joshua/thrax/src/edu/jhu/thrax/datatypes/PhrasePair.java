package edu.jhu.thrax.datatypes;

import java.util.Iterator;

import edu.jhu.thrax.extraction.SpanLabeler;

/**
 * This class represents a phrase pair. Essentially it is four integers
 * describing the boundaries of the source and target sides of the phrase pair.
 */
public class PhrasePair 
{
    /**
     * The index of the start of the source side of this PhrasePair.
     */
    public final int sourceStart;
    /**
     * One plus the index of the end of the source side of this PhrasePair.
     */
    public final int sourceEnd;
    /**
     * The index of the start of the target side of this PhrasePair.
     */
    public final int targetStart;
    /**
     * One plus the index of the end of the target side of this PhrasePair.
     */
    public final int targetEnd;

    /**
     * Constructor.
     *
     * @param ss source start
     * @param se source end
     * @param ts target start
     * @param te target end
     */
    public PhrasePair(int ss, int se, int ts, int te)
    {
        sourceStart = ss;
        sourceEnd = se;
        targetStart = ts;
        targetEnd = te;
    }

	/**
	 * Determines if another PhrasePair is contained (non-strictly) within
	 * this PhrasePair. Another PhrasePair is contained non-strictly if none
	 * of its boundary points lie outside of this PhrasePair.
	 *
	 * @param other the other PhrasePair
	 * @return true if other is contained non-strictly in this PhrasePar,
	 * false if at least one point lies outside
	 */
	public boolean contains(PhrasePair other)
	{
		return other.sourceStart >= sourceStart
			&& other.sourceEnd <= sourceEnd
			&& other.targetStart >= targetStart
			&& other.targetEnd <= targetEnd;
	}

	/**
	 * Determine if this PhrasePair can be considered as an initial phrase
	 * pair according to a particular alignment. A phrase pair is called an
	 * initial phrase pair if the following conditions are satisfied:
	 * <p>
	 * 1) no source words are aligned outside the target span of the phrase
	 * 2) no target words are aligned outside the source span of the phrase
	 * 3) a certain number of alignment points are present in the phrase pair
	 * <p>
	 * In addition, we may optionally specify that only the smallest phrase
	 * pair with the same alignment is kept: that is, we may disallow the
	 * presence of unaligned words at the edges of the PhrasePair.
	 *
	 * @param a the Alignment
	 * @param allowUnaligned whether to allow unaligned words at the edges of
	 * initial phrase pairs
	 * @param minimumAligned the minimum number of alignment points needed
	 * @return true if this is an initial phrase pair, false otherwise
	 */
	public boolean isInitialPhrasePair(Alignment a, boolean allowUnaligned, int minimumAligned)
	{
		int numLinks = 0;
		for (int i = sourceStart; i < sourceEnd; i++) {
			Iterator<Integer> js = a.targetIndicesAlignedTo(i);
			while (js.hasNext()) {
				numLinks++;
				int j = js.next();
				if (j < targetStart || j >= targetEnd)
					return false;
			}
		}
		for (int j = targetStart; j < targetEnd; j++) {
			Iterator<Integer> is = a.sourceIndicesAlignedTo(j);
			while (is.hasNext()) {
				numLinks++;
				int i = is.next();
				if (i < sourceStart || i >= sourceEnd)
					return false;
			}
		}
		return numLinks >= minimumAligned && (allowUnaligned || isMinimal(a));
	}

	private boolean isMinimal(Alignment a)
	{
		return a.sourceIndexIsAligned(sourceStart)
			&& a.sourceIndexIsAligned(sourceEnd - 1)
			&& a.targetIndexIsAligned(targetStart)
			&& a.targetIndexIsAligned(targetEnd - 1);
	}

	public int sourceLength()
	{
		return sourceEnd - sourceStart;
	}

	public int targetLength()
	{
		return targetEnd - targetStart;
	}

	public int numAlignmentPoints(Alignment a)
	{
		if (sourceLength() < targetLength())
			return countAlignmentPointsSource(a);
		else
			return countAlignmentPointsTarget(a);
	}

	private int countAlignmentPointsSource(Alignment a)
	{
		int result = 0;
		for (int i = sourceStart; i < sourceEnd; i++)
			result += a.numTargetWordsAlignedTo(i);
		return result;
	}

	private int countAlignmentPointsTarget(Alignment a)
	{
		int result = 0;
		for (int j = targetStart; j < targetEnd; j++)
			result += a.numSourceWordsAlignedTo(j);
		return result;
	}

    public String toString()
    {
        return String.format("[%d,%d)+[%d,%d)", sourceStart, sourceEnd, targetStart, targetEnd);
    }

    public boolean equals(Object o)
    {
        if (this == o)
            return true;
        if (!(o instanceof PhrasePair))
            return false;
        PhrasePair p = (PhrasePair) o;
        return sourceStart == p.sourceStart
            && sourceEnd == p.sourceEnd
            && targetStart == p.targetStart
            && targetEnd == p.targetEnd;
    }

    public int hashCode()
    {
        int result = 37;
        result *= 163 + sourceStart;
        result *= 163 + sourceEnd;
        result *= 163 + targetStart;
        result *= 163 + targetEnd;
        return result;
    }

	public int getLabel(SpanLabeler labeler, boolean useSource)
	{
		if (useSource)
			return labeler.getLabel(sourceStart, sourceEnd);
		else
			return labeler.getLabel(targetStart, targetEnd);
	}

	public boolean sourceIsDisjointFrom(PhrasePair other)
	{
		if (other.sourceStart < sourceStart)
			return other.sourceEnd <= sourceStart;
		return other.sourceStart >= sourceEnd;
	}

	public boolean targetIsDisjointFrom(PhrasePair other)
	{
		if (other.targetStart < targetStart)
			return other.targetEnd <= targetStart;
		return other.targetStart >= targetEnd;
	}
}
