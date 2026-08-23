package kits.ml.application.digitdetection;

import java.io.File;
import java.io.IOException;
import java.util.List;

import kits.ml.util.Logger;

public class DigitDetectorMain {

    public static void main(String[] args) throws IOException {
        
        //NeuralNetDigitDetector detector = new NeuralNetDigitDetector();
        for(int k : List.of(3)) {
            KNearestNeighbourDigitDetector detector = new KNearestNeighbourDigitDetector(k);
            File trainingSet = new File("Kaggle/mnist_png/training");
            detector.learn(trainingSet);
            
            File testSet = new File("Kaggle/mnist_png/testing");

            Logger.log("Testing with " + testSet);
            double result = test(detector, testSet);
            Logger.log("Result: " + String.format("%.2f%%", result * 100) + " k = " + k);
        }
        
    }

    private static double test(KNearestNeighbourDigitDetector detector, File testFolder) throws IOException {
        
        int count = 0;
        int success = 0;
        for(File digitFolder : testFolder.listFiles()) {
            int digit = Integer.parseInt(digitFolder.getName());
            for(File digitFile : digitFolder.listFiles()) {
                int detectedDigit = detector.detect(digitFile);
                count++;
                if(detectedDigit == digit) success++;
            }
        }
        
        return success / (double)count;
    }

}
