package kits.ml.core;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.stream.Collectors;

import org.junit.jupiter.api.Test;

@SuppressWarnings("static-method")
public class LogisticRegressionTest {

    private static final double TOLERANCE = 0.001;
    
    @Test
    public void test1() {

        LogisticRegressionModel model = new LogisticRegressionModel(2);

        List<LearningData> learningDataSet = FileReader.readLearningDataSet("input/ExamData.txt");

        model.learn(learningDataSet);
        
        assertEquals(0.6241, model.calculateOutput(new Input(45, 85)), TOLERANCE);
        assertEquals(0.5295, model.calculateOutput(new Input(11, 20)), TOLERANCE);
        assertEquals(0.7382, model.calculateOutput(new Input(95, 97)), TOLERANCE);
    }

    @Test
    public void test2() {

        LogisticRegressionModel model = new LogisticRegressionModel(28);

        List<LearningData> learningDataSet = FileReader.readLearningDataSet("input/MicroChipData.txt");

        List<LearningData> learningDataSet2 = learningDataSet.stream().map(LogisticRegressionTest::mapFeature).collect(Collectors.toList());

        System.out.println(model.calculateCost(learningDataSet2));

        model.learn(learningDataSet2);
    }

    private static LearningData mapFeature(LearningData learningData) {
        double[] values = new double[28];

        int c = 0;
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j <= i; j++) {
                values[c++] = Math.pow(learningData.input().get(0), i) * Math.pow(learningData.input().get(0), j);
            }
        }
        return new LearningData(new Input(values), learningData.output());
    }

}
