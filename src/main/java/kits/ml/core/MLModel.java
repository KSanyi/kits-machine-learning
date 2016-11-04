package kits.ml.core;

import java.util.Set;

public interface MLModel {

	void learn(Set<LearningData> learningDataSet);

	double calculateOutput(Input input);

	double calculateCost(Set<LearningData> learningDataSet);

}
