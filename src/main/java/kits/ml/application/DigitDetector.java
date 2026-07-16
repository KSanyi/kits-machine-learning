package kits.ml.application;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import kits.ml.core.DataPoint;
import kits.ml.core.math.MLMath;
import kits.ml.core.math.linalg.Vector;
import kits.ml.neuralnet.ActivationFunction.StandardActivationFunction;
import kits.ml.neuralnet.CostFunction.StandardCostFunction;
import kits.ml.neuralnet.NeuralNet;
import kits.ml.util.Logger;

public class DigitDetector {
    
    private static final Logger LOGGER = new Logger();
    
    private final NeuralNet neuralNet = new NeuralNet(StandardCostFunction.CATEGORICAL_CROSS_ENTROPY, StandardActivationFunction.RELU, StandardActivationFunction.SOFTMAX, 784, 256, 128, 10);

    public void learn(File trainingSet) throws IOException {
        List<DataPoint> dataPoints = loadDataPoints(trainingSet, 100);

        neuralNet.setNumberOfEpochs(100);
        neuralNet.setLearningRate(0.01);
        neuralNet.setLearningRateDecay(33, 0.5);
        
        neuralNet.learn(dataPoints);
    }

    private static List<DataPoint> loadDataPoints(File trainingFilesRoot, int dataPointsPerDigit) throws IOException {
        List<DataPoint> datapoints = new ArrayList<>();
        for(File digitFolder : trainingFilesRoot.listFiles()) {
            LOGGER.log("Reading folder " + digitFolder);
            int digit = Integer.parseInt(digitFolder.getName());
            int count = 0;
            for(File digitFile : digitFolder.listFiles()) {
                Vector input = createInputVector(digitFile);
                Vector output = MLMath.oneHot(10, digit);
                datapoints.add(new DataPoint(input, output));
                count++;
                if(count == dataPointsPerDigit) break;
            }
        }
        
        return datapoints;
    }
    
    private static Vector createInputVector(File digitFile) throws IOException {
        BufferedImage img = ImageIO.read(digitFile);
        
        int width = img.getWidth();
        int height = img.getHeight();

        double[] data = new double[height * width];
        int index = 0;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = img.getRGB(x, y);
                int greyScale = convertToGreyScale(rgb);
                data[index++] = greyScale / 255.0 - 0.5;
            }
        }
        
        return new Vector(data);
    }
    
    private static int convertToGreyScale(int rgb) {
        int r = (rgb >> 16) & 0xFF;
        int g = (rgb >> 8) & 0xFF;
        int b = rgb & 0xFF;
        return (r + g + b) / 3;
    }

    public int detect(File digitFile) throws IOException {
        Vector output = neuralNet.predict(createInputVector(digitFile));
        
        return convertoToDigit(output);
    }
    
    private static int convertoToDigit(Vector output) {
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

}
