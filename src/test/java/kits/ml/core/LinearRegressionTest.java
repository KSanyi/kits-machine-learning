package kits.ml.core;

import java.util.Arrays;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

public class LinearRegressionTest {

    @Test
    public void test1() {

        MLModel model = new LinearRegressionModel(1);

        List<LearningData> learningDataSet = Arrays.asList(
                new LearningData(new Input(0), 1), 
                new LearningData(new Input(1), 3), 
                new LearningData(new Input(2), 5),
                new LearningData(new Input(10), 21));

        model.learn(learningDataSet);

        final double DELTA = 0.001;
        Assert.assertEquals(2, model.calculateOutput(new Input(0.5)), DELTA);
        Assert.assertEquals(21, model.calculateOutput(new Input(10)), DELTA);
    }

    @Test
    public void test2() {

        MLModel model = new LinearRegressionModel(1, 0.1, 100);

        List<LearningData> learningDataSet = FileReader.readLearningDataSet("input/HousingData1.txt");

        model.learn(learningDataSet);

        double cost = model.calculateCost(learningDataSet);
        double prediction1 = model.calculateOutput(new Input(3.5));
        double prediction2 = model.calculateOutput(new Input(7));

        final double DELTA = 0.001;
        Assert.assertEquals(4.47697, cost, DELTA);
        Assert.assertEquals(0.27983, prediction1, DELTA);
        Assert.assertEquals(4.45545, prediction2, DELTA);
    }

    @Test
    public void test3() {

        MLModel model = new LinearRegressionModel(2);

        List<LearningData> learningDataSet = FileReader.readLearningDataSet("input/HousingData2.txt");

        model.learn(learningDataSet);

        double cost = model.calculateCost(learningDataSet);
        double prediction = model.calculateOutput(new Input(1650, 3));

        final double DELTA = 0.01;
        Assert.assertEquals(2.043280050602829E9, cost, 2.043280050602829E9 * DELTA);
        Assert.assertEquals(293081, prediction, 293081 * DELTA);
    }

}
