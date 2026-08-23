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
    
    private final CostFunction costFunction;
    
    private final ActivationFunction activationFunction;
    
    private double learningRate = 0.01;
    
    private double numberOfEpochs = 100;
    
    private double lambda = 0;
    
    public NeuralNet(CostFunction costFunction, ActivationFunction activationFunction, int ... neuronsPerLayer) {
        
        this.costFunction = costFunction;
        this.activationFunction = activationFunction;
        numberOfLayers = neuronsPerLayer.length;
        
        weightMatrixes = new Matrix[numberOfLayers-1];
        biasVectors = new Vector[numberOfLayers-1];
        
        for(int i=0;i<numberOfLayers-1;i++) {
            int numberOfNeuronsInCurrentLayer = neuronsPerLayer[i];
            int numberOfNeuronsInNextLayer = neuronsPerLayer[i+1];
            
            weightMatrixes[i] = new Matrix(numberOfNeuronsInNextLayer, numberOfNeuronsInCurrentLayer);
            biasVectors[i] = new Vector(numberOfNeuronsInNextLayer);
        }
        
        Logger.log("Neural net created with neurons per layer: " + Arrays.toString(neuronsPerLayer));
        
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
            Logger.log("Cost at epoch " + i + ": " + cost(dataPoints));
        }
    }
    
    public void learn(DataPoint dataPoint) {

        Vector[] activations = new Vector[numberOfLayers];
        activations[0] = dataPoint.input();
        int i;
        for(i=0;i<numberOfLayers-2;i++) {
            activations[i+1] = weightMatrixes[i].multiply(activations[i]).plus(biasVectors[i]).map(activationFunction::apply);
        }
        activations[i+1] = MLMath.softMax(weightMatrixes[i].multiply(activations[i]).plus(biasVectors[i]));

        Vector target = dataPoint.output();
        Gradient gradient = calculateGradient(activations, target);
        Logger.log(gradient.dWs[1].norm() + "");
        for(i=0;i<numberOfLayers-1;i++) {
            weightMatrixes[i] = weightMatrixes[i].minus(gradient.dWs[i].scale(learningRate));
            biasVectors[i] = biasVectors[i].minus(gradient.dbs[i].scale(learningRate));
        }
    }
    
    private Gradient calculateGradient(Vector[] activations, Vector target) {
        int n = weightMatrixes.length;
        Matrix[] dWs = new Matrix[n];
        Vector[] dbs = new Vector[n];
        Vector costGradient = costFunction.gradient(activations[n], target);
        Vector outputActivationGradient = activations[n].map(activationFunction::derivative);
        dbs[n-1] = costGradient.hadamardProduct(outputActivationGradient);
        dWs[n-1] = dbs[n-1].multiply(activations[n-1]).plus(weightMatrixes[n-1].scale(lambda));

        for (int i = n-2; i >= 0; i--) {
            Vector wTDelta = weightMatrixes[i+1].transpose().multiply(dbs[i+1]);
            Vector activationGrad = activations[i+1].map(activationFunction::derivative);
            dbs[i] = wTDelta.hadamardProduct(activationGrad);
            dWs[i] = dbs[i].multiply(activations[i]).plus(weightMatrixes[i].scale(lambda));
        }
        
        return new Gradient(dWs, dbs);
    }

    public Vector predict(Vector input) {
        Vector vector = input;
        int i;
        for(i=0;i<numberOfLayers-2;i++) {
            vector = weightMatrixes[i].multiply(vector).plus(biasVectors[i]).map(activationFunction::apply);
        }
        vector = MLMath.softMax(weightMatrixes[i].multiply(vector).plus(biasVectors[i]));
        return vector;
    }

    public double cost(List<DataPoint> dataPoints) {
        return dataPoints.stream().mapToDouble(this::cost).sum() / dataPoints.size();
    }
    
    public double cost(DataPoint dataPoint) {
        Vector predictedOutput = predict(dataPoint.input());
        Vector trueOutput = dataPoint.output();
        
        return costFunction.cost(predictedOutput, trueOutput);
    }

    public void setNumberOfEpochs(int numberOfEpochs) {
        this.numberOfEpochs = numberOfEpochs;
        Logger.log("Number of epochs rate is set to " + numberOfEpochs);
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
