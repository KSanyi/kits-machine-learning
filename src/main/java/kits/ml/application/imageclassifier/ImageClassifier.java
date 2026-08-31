package kits.ml.application.imageclassifier;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

import javax.imageio.ImageIO;

import kits.ml.core.DataPoint;
import kits.ml.core.math.MLMath;
import kits.ml.core.math.linalg.Vector;
import kits.ml.neuralnet.ActivationFunction.StandardActivationFunction;
import kits.ml.neuralnet.NeuralNet;
import kits.ml.util.Logger;

public class ImageClassifier {

    private final NeuralNet neuralNet;
    
    private List<String> labels = List.of();
    
    public ImageClassifier(int numberOfEpochs) {
        neuralNet = new NeuralNet(StandardActivationFunction.RELU, 3072, 100, 10);
        
        neuralNet.setNumberOfEpochs(numberOfEpochs);
        neuralNet.setLearningRate(0.01);
        neuralNet.setLambda(0.001);
    }
    
    public void train(List<ImageLearningData> trainingData) {
        
        labels = trainingData.stream().map(ImageLearningData::label).distinct().sorted().toList();
        
        Logger.log("Vectorizing images");
        
        List<DataPoint> datapoints = trainingData.stream()
                .map(d -> new DataPoint(createInputVector(d.file()), createOutputVector(d.label())))
                .collect(Collectors.toList());
        
        Logger.log("Learning starts");
        neuralNet.learn(datapoints);
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
        int index = MLMath.findMaxIndex(outputVector);
        return labels.get(index);
    }

    record ImageLearningData(File file, String label) {}
}
