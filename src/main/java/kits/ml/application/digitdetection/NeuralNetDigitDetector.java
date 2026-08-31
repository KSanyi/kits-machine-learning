package kits.ml.application.digitdetection;

import java.io.File;
import java.io.IOException;
import java.util.List;

import kits.ml.core.DataPoint;
import kits.ml.core.math.linalg.Vector;
import kits.ml.neuralnet.ActivationFunction.StandardActivationFunction;
import kits.ml.neuralnet.NeuralNet;

public class NeuralNetDigitDetector extends BaseDigitDetector {
    
    private final NeuralNet neuralNet = new NeuralNet(StandardActivationFunction.RELU, 784, 60, 10);
    
    public void learn(File trainingSet) throws IOException {
        List<DataPoint> dataPoints = loadDataPoints(trainingSet, 100);
        
        neuralNet.setNumberOfEpochs(100);
        neuralNet.setLearningRate(0.01);
        
        neuralNet.learn(dataPoints);
    }

    public int detect(File digitFile) throws IOException {
        Vector output = neuralNet.predict(createInputVector(digitFile));
        
        return convertoToDigit(output);
    }
    
}
