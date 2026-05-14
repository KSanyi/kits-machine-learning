package kits.ml.core;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

import kits.ml.core.math.linalg.Vector;
import kits.ml.core.math.optimization.GradientDescentOptimizer;

public class LinearRegressionTest {

    private static final double TOLERANCE = 0.01;
    
    @Test
    public void test1() {

        MLModel model = new LinearRegressionModel(1, new GradientDescentOptimizer(100, 0.05, 0.0001));

        List<LearningData> learningDataSet = Arrays.asList(
                new LearningData(new Vector( 0),  1), 
                new LearningData(new Vector( 1),  3), 
                new LearningData(new Vector( 2),  5),
                new LearningData(new Vector(10), 21));

        model.learn(learningDataSet);

        assertEquals(2, model.calculateOutput(new Vector(0.5)), TOLERANCE);
        assertEquals(201, model.calculateOutput(new Vector(100)), TOLERANCE);
    }

    @Test
    public void test2() {

        MLModel model = new LinearRegressionModel(1);

        List<LearningData> learningDataSet = FileReader.readLearningDataSet("input/HousingData1.txt");

        model.learn(learningDataSet);

        double cost = model.calculateCost(learningDataSet);
        double prediction1 = model.calculateOutput(new Vector(3.5));
        double prediction2 = model.calculateOutput(new Vector(7));

        assertEquals(4.47697, cost, TOLERANCE);
        assertEquals(0.27983, prediction1, TOLERANCE);
        assertEquals(4.45545, prediction2, TOLERANCE);
    }

    @Test
    public void test3() {

        MLModel model = new LinearRegressionModel(2);

        List<LearningData> learningDataSet = FileReader.readLearningDataSet("input/HousingData2.txt");

        model.learn(learningDataSet);

        double cost = model.calculateCost(learningDataSet);
        double prediction = model.calculateOutput(new Vector(1650, 3));

        assertEquals(2.043280050602829E9, cost, 2.043280050602829E9 * TOLERANCE);
        assertEquals(293081, prediction, 293081 * TOLERANCE);
    }

}
