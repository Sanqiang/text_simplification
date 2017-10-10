package edu.jhu.thrax.hadoop.features.pivot;

import java.util.Set;

import org.apache.hadoop.io.Writable;

import edu.jhu.thrax.hadoop.datatypes.FeatureMap;
import edu.jhu.thrax.hadoop.features.Feature;

public interface PivotedFeature extends Feature {

	public Set<String> getPrerequisites();

	public Writable pivot(FeatureMap src, FeatureMap tgt);

	public void initializeAggregation();
	
	public void aggregate(FeatureMap a);

	public Writable finalizeAggregation();
	
	public Set<String> getLowerBoundLabels();
	
	public Set<String> getUpperBoundLabels();
}
