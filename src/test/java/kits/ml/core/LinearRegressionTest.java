package kits.ml.core;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import org.junit.Test;

public class LinearRegressionTest {

	@Test
	public void test() {
		
		MLModel model = new LinearRegressionModel(1);
		
		Set<LearningData> learningDataSet = new HashSet<>(Arrays.asList(new LearningData(new Input(0), 1),
																		new LearningData(new Input(1), 3),
																		new LearningData(new Input(2), 5),
																		new LearningData(new Input(10), 21)));
		
		model.learn(learningDataSet);

		System.out.println(model.calculateOutput(new Input(10)));
		
	}
	
}
