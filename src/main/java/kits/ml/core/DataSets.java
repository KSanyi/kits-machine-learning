package kits.ml.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public record DataSets<T>(List<T> trainingData, List<T> testData) {

    public static <T> DataSets<T> create(List<T> allData, int trainingPercent) {
        
        Random random = new Random(0);
        
        List<T> trainingData = new ArrayList<>();
        List<T> testData = new ArrayList<>();
        
        for(T data : allData) {
            if(random.nextDouble() * 100 < trainingPercent) {
                trainingData.add(data);
            } else {
                testData.add(data);
            }
        }
        return new DataSets<T>(trainingData, testData);
    }
    
}
