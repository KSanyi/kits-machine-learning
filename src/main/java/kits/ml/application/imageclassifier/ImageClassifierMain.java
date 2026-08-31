package kits.ml.application.imageclassifier;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import kits.ml.application.imageclassifier.ImageClassifier.ImageLearningData;
import kits.ml.core.DataSets;
import kits.ml.util.Logger;

public class ImageClassifierMain {

    public static void main(String[] args) throws IOException {
        
        DataSets<ImageLearningData> datasets = createDataSets(90);
        ImageClassifier imageClassifier = new ImageClassifier(10);
        List<ImageLearningData> trainingData = datasets.trainingData();//sample(datasets.trainingData(), 1000);
        for(int i=0;i<20;i++) {
            Logger.log("Training round " + (i+1) + " starts");
            imageClassifier.train(trainingData);
            
            Logger.log("Result on training set: " + test(imageClassifier, trainingData));
            Logger.log("Result on test set: " + test(imageClassifier, datasets.testData()));
        }
        
    }

    private static List<ImageLearningData> sample(List<ImageLearningData> trainingData, int numberOfSamples) {
        Random random = new Random(0);
        List<ImageLearningData> copy = new ArrayList<>(trainingData);
        List<ImageLearningData> samples = new ArrayList<>(numberOfSamples);
        for(int i=0;i<numberOfSamples;i++) {
            int index = random.nextInt(copy.size());
            samples.add(copy.get(index));
            copy.remove(index);
        }
        return samples;
    }

    private static DataSets<ImageLearningData> createDataSets(int trainingPercent) throws IOException {
        File trainingSet = new File("Kaggle/cifar/train");
        List<String> lines = Files.readAllLines(Paths.get("Kaggle/cifar/trainLabels.csv"));
        Map<String, String> fileToLabel = new HashMap<>();
        for(int i=1;i<lines.size();i++) {
            String line = lines.get(i);
            String parts[] = line.split(",");
            fileToLabel.put(parts[0], parts[1]);
        }
        
        List<String> labels = fileToLabel.values().stream().distinct().sorted().toList();
        Logger.log("Labels in the dataset: " + labels);
        
        List<ImageLearningData> allData = new ArrayList<>();
        for(File file : trainingSet.listFiles()) {
            String fileName = file.getName();
            String fileNameWithoutExtension = fileName.substring(0, fileName.length()-4); // strip .png
            String label = fileToLabel.get(fileNameWithoutExtension);
            
            allData.add(new ImageLearningData(file, label));
        }
        
        return DataSets.create(allData, trainingPercent);
    }
    
    private static double test(ImageClassifier imageClassifier, List<ImageLearningData> testData) throws IOException {
        int count = 0;
        int success = 0;
        for(ImageLearningData dataPoint : testData) {
            String predictedLabel = imageClassifier.predict(dataPoint.file());
            count++;
            if(predictedLabel.equals(dataPoint.label())) success++;
        }
        
        return success / (double)count;
    }
    
}
