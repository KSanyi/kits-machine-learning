package kits.ml.core;

import java.util.Set;

public class Main {

	public static void main(String[] args) {

		MLModel model = new LinearRegressionModel(2);

		Set<LearningData> learningDataSet = FileReader.readLearningDataSet("input/learningdata");
		
		model.learn(learningDataSet);
		
		Input input = FileReader.readinput("input/data");
		
		double output = model.calculateOutput(input);
		
		double cost = model.calculateCost(learningDataSet);
	}

}
