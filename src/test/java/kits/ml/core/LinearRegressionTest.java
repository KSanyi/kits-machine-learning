package kits.ml.core;

import java.util.Arrays;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

public class LinearRegressionTest {

	private final double EPSILON = 0.0001;
	
	@Test
	public void test() {
		
		MLModel model = new LinearRegressionModel(1);
		
		List<LearningData> learningDataSet = Arrays.asList(new LearningData(new Input(0), 1),
																		new LearningData(new Input(1), 3),
																		new LearningData(new Input(2), 5),
																		new LearningData(new Input(10), 21));
		
		model.learn(learningDataSet);

		Assert.assertEquals(2, model.calculateOutput(new Input(0.5)), EPSILON);
		Assert.assertEquals(21, model.calculateOutput(new Input(10)), EPSILON);
	}
	
}
