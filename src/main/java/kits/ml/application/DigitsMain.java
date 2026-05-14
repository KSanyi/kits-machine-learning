package kits.ml.application;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import javax.imageio.ImageIO;

import kits.ml.core.DataSets;
import kits.ml.core.LearningData;
import kits.ml.core.math.linalg.Vector;
import kits.ml.neuralnet.VectorizedNeuralNet;

public class DigitsMain {

    public static void main(String[] args) throws Exception {
        
         VectorizedNeuralNet neuralNet = teachNeuralNet();
         //neuralNet.save("./digitsnet");
        
//        VectorizedNeuralNet neuralNet = VectorizedNeuralNet.load("./digitsnet");
//        
//        for(File folder : new File("C:\\projects\\kits-machine-learning\\Kaggle\\Digits").listFiles()) {
//            for(File file : folder.listFiles()) {
//                Vector input = createInput(file);
//                System.out.println(file.getName() + " => " + findIndexForMaxOutput(neuralNet.predict(input)));
//            }
//        }
        
    }
    
    private static VectorizedNeuralNet teachNeuralNet() throws Exception {
        System.out.println("Start");
        List<LearningData> learningData = createLearningData("Kaggle/Digits");
        
        System.out.println("Splitting data to training and test sets");
        DataSets dataSets = DataSets.create(learningData, 80);
        
        int inputDimenstion = dataSets.testData().get(0).input().length();
        
        VectorizedNeuralNet neuralNet = new VectorizedNeuralNet(0.001, inputDimenstion, 30, 10);
        learn(neuralNet, dataSets.trainingData());
        
        return neuralNet;
    }
    
    private static double calculateCost(VectorizedNeuralNet neuralNet, List<LearningData> trainingData) {
        return trainingData.stream()
                .mapToDouble(data -> calculateCost(neuralNet, data.input(), data.output()))
                .sum() / trainingData.size();
    }

    private static double calculateCost(VectorizedNeuralNet neuralNet, Vector input, double output) {
        Vector trueOutputVector = Vector.createOneHot(10, (int)output);
        Vector outputVector = neuralNet.predict(input);
        return outputVector.minus(trueOutputVector).normSquared();
    }

    private static List<LearningData> createLearningData(String path) throws Exception {
        
        System.out.println("Loading data from " + path);
        List<LearningData> learningDatas = new ArrayList<>();
        for(File folder : Paths.get(path).toFile().listFiles()) {
            String name = folder.getName();
            int digit = Integer.parseInt(name);
            for(File file : folder.listFiles()) {
                Vector input = createInput(file);
                learningDatas.add(new LearningData(input, digit));
            }
            System.out.println("Data loaded for folder " + name);
        }
        
        return learningDatas;
    }
    
    private static Vector createInput(File file) throws Exception {
        BufferedImage img = ImageIO.read(file);

        int width = img.getWidth();
        int height = img.getHeight();

        double[] data = new double[height * width];
        int index = 0;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {

                int rgb = img.getRGB(x, y);

                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                // convert to grayscale
                int gray = (r + g + b) / 3; // or: (int)(0.299 * r + 0.587 * g + 0.114 * b);

                data[index++] = (gray / 255.0 - 0.5) / 2;
            }
        }
        
        return new Vector(data);
    }
    
    private static void learn(VectorizedNeuralNet neuralNet, List<LearningData> trainingData) {
        neuralNet.randomizeWeights();

        Collections.shuffle(trainingData);
        
        System.out.println("Initial cost: " + calculateCost(neuralNet, trainingData));
        for(int i=0;i<100;i++) {
            for(LearningData learningData : trainingData) {
                neuralNet.learn(learningData.input(), Vector.createOneHot(10, (int)learningData.output()));
            }
            // System.out.println("Epoch " + (i + 1) + " cost: " + calculateCost(neuralNet, trainingData));
            // test(neuralNet, trainingData);
            System.out.println("Epoch " + i);
        }
    }
    
    private static void test(VectorizedNeuralNet neuralNet, List<LearningData> testData) {
        int score = 0;
        for(LearningData testDataRow : testData) {
            int predictedNumber = findIndexForMaxOutput(neuralNet.predict(testDataRow.input()));
            if(predictedNumber == testDataRow.output()) score++;
            //System.out.println(testDataRow.output() + " predicted: " + predictedNumber);
        }
        System.out.println((double)score / testData.size());
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
