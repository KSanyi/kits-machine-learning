package kits.ml.coursera;

import java.util.ArrayList;
import java.util.List;

import kits.ml.core.Input;
import kits.ml.core.LearningData;
import kits.ml.core.LogisticRegressionModel;
import kits.ml.core.math.MLMath;
import kits.ml.core.math.linalg.Vector;

public class Exercise2 {

    static final double[][] X = new double[][] {
        {1,  2.2873553,   0.8908079},
        {1,  2.4717267,  -0.6861101},
        {1,  0.3836040,  -1.6322217},
        {1, -2.0572025,  -1.0776761},
        {1, -2.6066264,   0.4676799},
        {1, -0.7595301,   1.5830532},
        {1,  1.7858747,   1.2429747},
        {1,  2.6893545,  -0.2398890},
        {1,  1.1202542,  -1.5021998},
        {1, -1.4788027,  -1.3833951},
        {1, -2.7182552,   0.0072967},
        {1, -1.4585564,   1.3912800},
        {1,  1.1421324,   1.4961268},
        {1,  2.6927500,   0.2254416},
        {1,  1.7676656,  -1.2525136},
        {1, -0.7826024,  -1.5789136},
        {1, -2.6133493,  -0.4536676},
        {1, -2.0413950,   1.0886782},
        {1,  0.4074085,   1.6300983},
        {1,  2.4816425,   0.6728136}};
        
    static double[] y = new double[20];
    
    static {
        for (int i = 0; i < 20; i++) {
            double testValue = Math.sin(X[i][0] + X[i][1]);
            y[i] = testValue > 0 ? 1 : 0;
        }
    }
         
    public static void main(String[] args) {
        task1();
        System.out.println();
        
        task2();
        System.out.println();
        
        task3();
    }
    
    private static void task1() {
        System.out.println("Task 1");
        System.out.println("Sigmoid values: ");
        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 20; i++) {
                System.out.format("%.5f ", MLMath.sigmoid(X[i][j]));
            }
        }
        System.out.println();
    }
    
    private static void task2() {
        System.out.println("Task 2");
        List<LearningData> learningDataSet = new ArrayList<>();
        
        for (int i = 0; i < 20; i++) {
            Input learningInput = new Input(X[i][1], X[i][2]);
            LearningData learningData = new LearningData(learningInput, y[i]);
            learningDataSet.add(learningData);
        }
        
        LogisticRegressionModel model = new LogisticRegressionModel(2);
        model.setParameters(0.25, 0.5, -0.5);
      
        System.out.println("Cost: " + model.calculateCost(learningDataSet));
        
        Vector gradient = model.calculateGradient(learningDataSet);
        
        System.out.println("Gradient: " + gradient);
        System.out.println();
        
        System.out.println("Predictions: ");
        learningDataSet.stream().forEach(data -> System.out.format("%.5f ", (double)model.predict(data.input())));
        System.out.println();
    }
    
    private static void task3() {
        System.out.println("Task 3");
        List<LearningData> learningDataSet = new ArrayList<>();
        
        for (int i = 0; i < 20; i++) {
            Input learningInput = new Input(X[i][1], X[i][2]);
            LearningData learningData = new LearningData(learningInput, y[i]);
            learningDataSet.add(learningData);
        }
        
        LogisticRegressionModel model = new LogisticRegressionModel(2, 100, 0.1);
        model.setParameters(0.25, 0.5, -0.5);
        
        System.out.println("Cost: " + model.calculateCost(learningDataSet));
        
        Vector gradient = model.calculateGradient(learningDataSet);
        
        System.out.println("Gradient: " + gradient);
    }

}
