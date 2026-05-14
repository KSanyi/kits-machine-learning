package kits.ml.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public record DataSets(List<LearningData> trainingData, List<LearningData> testData) {

    public static DataSets create(List<LearningData> allData, int trainingPercent) {
        
        Random random = new Random(0);
        
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
    
}
