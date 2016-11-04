package kits.ml.core;

import java.util.List;

public interface MLModel {

	void learn(List<LearningData> learningDataSet);

	double calculateOutput(Input input);

	double calculateCost(List<LearningData> learningDataSet);

}
