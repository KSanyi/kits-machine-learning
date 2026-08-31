package kits.ml.neuralnet;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import kits.ml.core.DataPoint;
import kits.ml.core.math.MLMath;
import kits.ml.core.math.linalg.Matrix;
import kits.ml.core.math.linalg.Vector;
import kits.ml.util.Logger;

public class NeuralNet {

    private final Matrix[] weightMatrixes;
    
    private final Vector[] biasVectors;
    
    private final int numberOfLayers;
    
    private final CostFunction costFunction = CostFunction.StandardCostFunction.CROSS_ENTROPY;
    
    private final ActivationFunction hiddenLayerActivationFunction; // output layer is always SoftMax
    
    private double learningRate = 0.01;
    
    private double numberOfEpochs = 100;
    
    private double lambda = 0;
    
    public NeuralNet(ActivationFunction hiddenLayerActivationFunction, int ... neuronsPerLayer) {
        
        this.hiddenLayerActivationFunction = hiddenLayerActivationFunction;
        numberOfLayers = neuronsPerLayer.length;
        
        weightMatrixes = new Matrix[numberOfLayers-1];
        biasVectors = new Vector[numberOfLayers-1];
        
        for(int i=0;i<numberOfLayers-1;i++) {
            int numberOfNeuronsInCurrentLayer = neuronsPerLayer[i];
            int numberOfNeuronsInNextLayer = neuronsPerLayer[i+1];
            
            weightMatrixes[i] = new Matrix(numberOfNeuronsInNextLayer, numberOfNeuronsInCurrentLayer);
            biasVectors[i] = new Vector(numberOfNeuronsInNextLayer);
        }
        Logger.log("Neural net created with neurons per layer: " + Arrays.toString(neuronsPerLayer) + ". Hidden layer activation function: " + hiddenLayerActivationFunction.name());
        
        randomizeWeights();
    }
    
    private void randomizeWeights() {
        Random random = new Random();
        for(Matrix matrix : weightMatrixes) {
            // Xavier initialization
            double sigma = MLMath.sqrt(1.0 / (matrix.getNrColumns() + matrix.getNrRows()));
            matrix.apply(x -> random.nextGaussian() * sigma);
        }
        
        for(Vector biasVector : biasVectors) {
            biasVector.apply(i -> random.nextDouble(-0.01, 0.01));
        }
    }

    public void learn(List<DataPoint> dataPoints) {
        for(int i=0;i<numberOfEpochs;i++) {
            Collections.shuffle(dataPoints);
            for(DataPoint datapoint : dataPoints) {
                learn(datapoint);
            }
            Logger.log("Learning done for epoch, checking cost");
            Logger.log("Cost at epoch " + i + ": " + cost(dataPoints));
        }
        
    }
    
    public void learn(DataPoint dataPoint) {

        Vector[] activations = forwardPass(dataPoint.input());

        Gradient gradient = calculateGradient(activations, dataPoint.output());
        for(int i=0;i<numberOfLayers-1;i++) {
            gradient.dWs[i].scaleThis(learningRate);
            gradient.dbs[i].scaleThis(learningRate);
            weightMatrixes[i].minusThis(gradient.dWs[i]);
            biasVectors[i].minusThis(gradient.dbs[i]);
        }
    }
    
    private Vector[] forwardPass(Vector input) {
        Vector[] activations = new Vector[numberOfLayers];
        activations[0] = input;
        int i;
        for(i=0;i<numberOfLayers-2;i++) {
            activations[i+1] = weightMatrixes[i].multiply(activations[i]).plus(biasVectors[i]).map(hiddenLayerActivationFunction::apply);
        }
        activations[i+1] = MLMath.softMax(weightMatrixes[i].multiply(activations[i]).plus(biasVectors[i]));
        return activations;
    }
    
    private Gradient calculateGradient(Vector[] activations, Vector target) {
        int n = weightMatrixes.length;
        Matrix[] dWs = new Matrix[n];
        Vector[] dbs = new Vector[n];
        dbs[n-1] = activations[n].minus(target);
        dWs[n-1] = dbs[n-1].multiply(activations[n-1]);

        for (int i = n-2; i >= 0; i--) {
            Vector wTDelta = weightMatrixes[i+1].transpose().multiply(dbs[i+1]);
            Vector activationGrad = activations[i+1].map(hiddenLayerActivationFunction::derivative);
            dbs[i] = wTDelta.hadamardProduct(activationGrad);
            dWs[i] = dbs[i].multiply(activations[i]);
        }
        
        if(lambda > 0) {
            for(int i=0;i<weightMatrixes.length;i++) { 
                dWs[i].plusThis(weightMatrixes[i].scale(lambda));
            } 
        }
        
        return new Gradient(dWs, dbs);
    }

    public Vector predict(Vector input) {
        Vector[] activations = forwardPass(input);
        return activations[activations.length-1];
    }

    public double cost(List<DataPoint> dataPoints) {
        double cost = dataPoints.stream().mapToDouble(this::cost).sum() / dataPoints.size();
        double regularisationCost = 0;
        if(lambda > 0) {
            for(Matrix w : weightMatrixes) {
                regularisationCost += w.allValuesStream().map(MLMath::square).sum();
            }
            cost += regularisationCost * lambda;
        }
        
        return cost;
    }
    
    public double cost(DataPoint dataPoint) {
        Vector predictedOutput = predict(dataPoint.input());
        Vector trueOutput = dataPoint.output();
        
        return costFunction.cost(predictedOutput, trueOutput);
    }

    public void setNumberOfEpochs(int numberOfEpochs) {
        this.numberOfEpochs = numberOfEpochs;
        Logger.log("Number of epochs is set to " + numberOfEpochs);
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        Logger.log("Learning rate is set to " + learningRate);
    }
    
    public void setLambda(double lambda) {
        this.lambda = lambda;
        Logger.log("Lambda is set to " + lambda);
    }
    
    private static record Gradient(Matrix[] dWs, Vector[] dbs) {}

}
