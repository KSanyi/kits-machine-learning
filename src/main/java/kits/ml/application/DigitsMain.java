package kits.ml.application;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import kits.ml.core.DataSets;
import kits.ml.core.LearningData;
import kits.ml.core.math.linalg.Vector;
import kits.ml.neuralnet.VectorizedNeuralNet;
import kits.ml.util.StopWatch;

public class DigitsMain {

    private static final Logger log = LoggerFactory.getLogger(DigitsMain.class);

    public static void main(String[] args) throws Exception {
        
        Random random = new Random(0);
        VectorizedNeuralNet neuralNet = teachNeuralNet(random);
//        neuralNet.save("./digitsnet");
        
//         VectorizedNeuralNet neuralNet = VectorizedNeuralNet.load("./digitsnet_20260514");
//         
//         neuralNet.save2("./digitsnet_20260514_2");
//        
//        
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
    
    private static VectorizedNeuralNet teachNeuralNet(Random random) throws Exception {
        log.info("Start");
        List<LearningData> learningData = createLearningData("Kaggle/Digits");
        
        log.info("Splitting data to training and test sets");
        DataSets dataSets = DataSets.create(learningData, 80);
        
        int inputDimenstion = dataSets.testData().get(0).input().length();
        
        //for(int hiddenLayers: List.of(16, 32, 64, 128)) {
            log.info("Using {} hidden layers", 32);
            VectorizedNeuralNet neuralNet = new VectorizedNeuralNet(0.004, random, inputDimenstion, 32, 10);
            learn(neuralNet, dataSets.trainingData(), random);
        //}
        
        return null;
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
        
        log.info("Loading data from {}", path);
        List<LearningData> learningDatas = new ArrayList<>();
        for(File folder : Paths.get(path).toFile().listFiles()) {
            String name = folder.getName();
            int digit = Integer.parseInt(name);
            for(File file : folder.listFiles()) {
                Vector input = createInput(file);
                learningDatas.add(new LearningData(input, digit));
            }
            log.info("Data loaded for folder {}", name);
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
    
    private static void learn(VectorizedNeuralNet neuralNet, List<LearningData> trainingData, Random random) {
        neuralNet.randomizeWeights();

        log.info("Initial cost: {}", calculateCost(neuralNet, trainingData));
        int n = 200;
        for(int i=0;i<n;i++) {
            train(neuralNet, trainingData, random);
            //System.out.println("Epoch " + (i + 1) + " cost: " + calculateCost(neuralNet, trainingData));
            //if(i % 10 == 0)
            test(neuralNet, trainingData);
        }
    }
    
    private static void train(VectorizedNeuralNet neuralNet, List<LearningData> trainingData, Random random) {
        Collections.shuffle(trainingData, random);
        
        List<List<LearningData>> batches = createBatches(trainingData, 4);
        for(List<LearningData> batch : batches) {
            neuralNet.learn(batch);
        }
    }
    
    private static List<List<LearningData>> createBatches(List<LearningData> trainingData, int batchSize) {
        List<List<LearningData>> batches = new ArrayList<List<LearningData>>();
        for(int i=0;i<trainingData.size()-batchSize;i+=batchSize) {
            batches.add(trainingData.subList(i, i+batchSize));
        }
        return batches;
    }

    private static void test(VectorizedNeuralNet neuralNet, List<LearningData> testData) {
        int score = 0;
        for(LearningData testDataRow : testData)  {
            int predictedNumber = findIndexForMaxOutput(neuralNet.predict(testDataRow.input()));
            if(predictedNumber == testDataRow.output()) score++;
            //System.out.println(testDataRow.output() + " predicted: " + predictedNumber);
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
