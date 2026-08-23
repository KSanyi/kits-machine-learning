package kits.ml.application.imageclassifier;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

import javax.imageio.ImageIO;

import kits.ml.core.DataPoint;
import kits.ml.core.math.linalg.Vector;
import kits.ml.neuralnet.ActivationFunction.StandardActivationFunction;
import kits.ml.neuralnet.CostFunction.StandardCostFunction;
import kits.ml.neuralnet.NeuralNet;
import kits.ml.util.Logger;

public class ImageClassifier {

    private final NeuralNet neuralNet;
    
    private List<String> labels = List.of();
    
    public ImageClassifier() {
        neuralNet = new NeuralNet(StandardCostFunction.CROSS_ENTROPY, StandardActivationFunction.RELU, 3072, 100, 10);
        
        neuralNet.setNumberOfEpochs(100);
        neuralNet.setLearningRate(0.001);
        //neuralNet.setLambda(0.001);
    }
    
    public void train(List<ImageLearningData> trainingData, List<ImageLearningData> testData) {
        
        labels = trainingData.stream().map(ImageLearningData::label).distinct().sorted().toList();
        
        Logger.log("Vectorizing images");
        
        List<DataPoint> datapoints = trainingData.stream()
                .map(d -> new DataPoint(createInputVector(d.file()), createOutputVector(d.label())))
                .collect(Collectors.toList());
        
        Logger.log("Learning starts");
        //for(int i=0;i<500;i++) {
            neuralNet.learn(datapoints);
        //    Logger.log("Epoch" + (i+1));
            //if(i > 30 || i % 5 == 0) {
        //        Logger.log("Result on training set: " +  test(trainingData));
        //        Logger.log("Result on test set: " +  test(testData));    
            //}
        //}
        
    }

    private static Vector createInputVector(File image) {
        BufferedImage img;
        try {
            img = ImageIO.read(image);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        
        int width = img.getWidth();
        int height = img.getHeight();
        int length = height * width;
        
        double[] dataR = new double[length];
        double[] dataG = new double[length];
        double[] dataB = new double[length];
        int index = 0;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = img.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                dataR[index] = r / 255.0 - 0.5;
                int g = (rgb >> 8) & 0xFF;
                dataG[index] = g / 255.0 - 0.5;
                int b = rgb & 0xFF;
                dataB[index] = b / 255.0 - 0.5;
                index++;
            }
        }
        
        double[] data = new double[length * 3];
        int pos = 0;
        System.arraycopy(dataR, 0, data, pos, length);
        pos += length;
        System.arraycopy(dataG, 0, data, pos, length);
        pos += length;
        System.arraycopy(dataB, 0, data, pos, length);
        
        return new Vector(data);
    }
    
    private Vector createOutputVector(String label) {
        int index = labels.indexOf(label);
        return Vector.createOneHot(labels.size(), index);
    }
    
    public String predict(File image) {
        Vector inputVector = createInputVector(image);
        Vector outputVector = neuralNet.predict(inputVector);
        int index = convertoToIndex(outputVector);
        return labels.get(index);
    }
    
    private static int convertoToIndex(Vector output) {
        int indexWithMaxValue = -1;
        double maxValue = -1;
        for(int i=0;i<output.length();i++) {
            if(output.get(i) > maxValue) {
                maxValue = output.get(i);
                indexWithMaxValue = i;
            }
        }
        return indexWithMaxValue;
    }
    
    private double test(List<ImageLearningData> testData) {
        int count = 0;
        int success = 0;
        for(ImageLearningData dataPoint : testData) {
            String predictedLabel = predict(dataPoint.file());
            count++;
            if(predictedLabel.equals(dataPoint.label())) success++;
        }
        
        return success / (double)count;
    }

    record ImageLearningData(File file, String label) {}
}
