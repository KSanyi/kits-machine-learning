package kits.ml.coursera;

import java.util.Arrays;
import java.util.List;

import kits.ml.core.Input;
import kits.ml.core.LearningData;
import kits.ml.core.MultiClassificationModel;

public class Exercise3 {

    static final List<LearningData> learningDataSet = Arrays.asList(
            new LearningData(new Input(-1, -1), 1),
            new LearningData(new Input(-1, -2), 1),
            new LearningData(new Input(-2, -1), 1),
            new LearningData(new Input(-2, -2), 1),
            new LearningData(new Input( 1,  1), 2),
            new LearningData(new Input( 1,  2), 2),
            new LearningData(new Input( 2,  1), 2),
            new LearningData(new Input( 2,  2), 2),
            new LearningData(new Input(-1,  1), 3),
            new LearningData(new Input(-1,  2), 3),
            new LearningData(new Input(-2,  1), 3),
            new LearningData(new Input(-2,  2), 3),
            new LearningData(new Input( 1, -1), 4),
            new LearningData(new Input( 1, -2), 4),
            new LearningData(new Input(-2, -1), 4),
            new LearningData(new Input(-2, -2), 4));
        
         
    public static void main(String[] args) {
        task1();
        System.out.println();
        task2();
    }
    
    
    private static void task1() {
        System.out.println("Task 1");
        
        MultiClassificationModel model = new MultiClassificationModel(2, 4, 0.1);
        
        model.learn(learningDataSet);
        System.out.println();
        
        learningDataSet.stream().forEach(learningData -> System.out.format("%.5f ", (double)model.predict(learningData.input)));
    }
    
    private static void task2() {
        System.out.println("Task 2");
        
        MultiClassificationModel model = new MultiClassificationModel(2, 4);
        
        double[][] params = new double[][] {
            { 0.84147,  0.41212, -0.96140},
            { 0.14112, -0.99999,  0.14988},
            {-0.95892,  0.42017,  0.83666},
            { 0.65699,  0.65029, -0.84622}  
        };
        model.setModelParams(params);
        
        learningDataSet.stream().forEach(learningData -> System.out.format("%.5f ", (double)model.predict(learningData.input)));
    }

}
