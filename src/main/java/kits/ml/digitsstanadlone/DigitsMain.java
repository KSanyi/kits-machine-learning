package kits.ml.digitsstanadlone;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DigitsMain {

    private static final Logger log = LoggerFactory.getLogger(DigitsMain.class);

    public static void main(String[] args) throws Exception {
        
        Random random = new Random(0);
        
        log.info("Start");
        List<LearningData> learningData = FileUtil.createLearningData("Kaggle/Digits");
        
        log.info("Splitting data to training and test sets");
        DataSets dataSets = createDataSets(learningData, 50, random);
        
        // VectorizedNeuralNet neuralNet = createNeuralNet(random, dataSets.trainingData());
        int inputDimenstion = learningData.get(0).input().length();
        VectorizedNeuralNet neuralNet = new VectorizedNeuralNet(0.004, new QuadraticCostFunction(), random, inputDimenstion, 30, 10);
        
        train(neuralNet, dataSets.trainingData(), random);
        //neuralNet.save("./digitsnet");
        
        test(neuralNet, dataSets.testData());
    }
    
    private static DataSets createDataSets(List<LearningData> allData, int trainingPercent, Random random) {
        
        List<LearningData> trainingData = new ArrayList<>();
        List<LearningData> testData = new ArrayList<>();
        
        for(LearningData data : allData) {
            if(random.nextDouble() * 100 < trainingPercent) {
                trainingData.add(data);
            } else {
                testData.add(data);
            }
        }
        return new DataSets(trainingData, testData);
    }
    
    private static void train(VectorizedNeuralNet neuralNet, List<LearningData> trainingData, Random random) {
        neuralNet.randomizeWeights();

        for(int i=0;i<50;i++) {
            Collections.shuffle(trainingData, random);
            
            for(LearningData data : trainingData) {
                neuralNet.learn(data);
            }
            log.info("Cost at epoch {}: {}", i, neuralNet.calculateCost(trainingData));
            test(neuralNet, trainingData);
        }
    }
    
    private static void test(VectorizedNeuralNet neuralNet, List<LearningData> testData) {
        int score = 0;
        for(LearningData testDataRow : testData)  {
            int predictedNumber = findIndexForMaxOutput(neuralNet.predict(testDataRow.input()));
            if(predictedNumber == testDataRow.output()) score++;
        }
        log.info("Score: {}", (double)score / testData.size());
    }
    
    private static int findIndexForMaxOutput(Vector vector) {
        double max = 0;
        int indexForMax = -1;
        for (int i = 0; i < vector.length(); i++) {
            if (vector.get(i) > max) {
                max = vector.get(i);
                indexForMax = i;
            }
        }
        return indexForMax;
    }
    
}
