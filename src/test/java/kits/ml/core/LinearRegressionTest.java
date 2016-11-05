package kits.ml.core;

import java.util.Arrays;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

public class LinearRegressionTest {

	private final double EPSILON = 0.001;
	
	//@Test
	public void test1() {
		
		MLModel model = new LinearRegressionModel(1);
		
		List<LearningData> learningDataSet = Arrays.asList(
				new LearningData(new Input(0), 1),
				new LearningData(new Input(1), 3),
				new LearningData(new Input(2), 5),
				new LearningData(new Input(10), 21));
		
		model.learn(learningDataSet);

		Assert.assertEquals(2, model.calculateOutput(new Input(0.5)), EPSILON);
		Assert.assertEquals(21, model.calculateOutput(new Input(10)), EPSILON);
	}
	
	@Test
	public void test2() {
		
		MLModel model = new LinearRegressionModel(1, 0.1, 100);
		
		List<LearningData> learningDataSet = FileReader.readLearningDataSet("input/HousingData1.txt");
		
		model.learn(learningDataSet);
		
		double cost = model.calculateCost(learningDataSet);
		double prediction1 = model.calculateOutput(new Input(3.5));
		double prediction2 = model.calculateOutput(new Input(7));
		
		//Assert.assertEquals(32.07, cost, EPSILON);
		Assert.assertEquals(0.45197, prediction1, EPSILON);
		Assert.assertEquals(4.53424, prediction2, EPSILON);
	}
	
	//@Test
	public void test3() {
		
		MLModel model = new LinearRegressionModel(2);
		
		List<LearningData> learningDataSet = FileReader.readLearningDataSet("input/HousingData2.txt");
		
		model.learn(learningDataSet);
		
		double cost = model.calculateCost(learningDataSet);
		double prediction = model.calculateOutput(new Input(1650, 3));
		
		//Assert.assertEquals(32.07, cost, EPSILON);
		Assert.assertEquals(32.07, prediction, EPSILON);

	}
	
}
