package kits.ml.application;

import java.io.File;
import java.io.IOException;

import kits.ml.util.Logger;

public class DigitDetectorMain {

    private static final Logger LOGGER = new Logger();
    
    public static void main(String[] args) throws IOException {
        
        DigitDetector detector = new DigitDetector();
        File trainingSet = new File("c:\\Users\\SandorKocso\\Documents\\neuralnets\\data\\mnist-png\\train\\");
        detector.learn(trainingSet);
        
        File testSet = new File("c:\\Users\\SandorKocso\\Documents\\neuralnets\\data\\mnist-png\\test\\");

        LOGGER.log("Testing with " + testSet);
        double result = test(detector, testSet);
        LOGGER.log("Result: " + String.format("%.2f%%", result * 100));
    }

    private static double test(DigitDetector detector, File testFolder) throws IOException {
        
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
