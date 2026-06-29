package kits.ml.neuralnet;

import java.util.Collections;
import java.util.List;
import java.util.Random;

import kits.ml.core.DataPoint;
import kits.ml.core.math.MLMath;
import kits.ml.core.math.linalg.Matrix;
import kits.ml.core.math.linalg.Vector;
import kits.ml.util.Logger;

public class NeuralNet {

    private final Logger logger = new Logger();

    private final Matrix[] weightMatrixes;

    private final Vector[] biasVectors;

    private final int numberOfLayers;

    private final CostFunction costFunction;

    private final ActivationFunction activationFunction;

    // output layer uses its own activation (must map to (0,1) for cross-entropy)
    private final ActivationFunction outputActivationFunction;

    private double learningRate = 0.01;

    private int numberOfEpochs = 100;

    private int lrDecaySteps = 0;   // 0 = disabled
    private double lrDecayFactor = 0.5;

    public NeuralNet(CostFunction costFunction, ActivationFunction activationFunction, int ... neuronsPerLayer) {
        this(costFunction, activationFunction, ActivationFunction.StandardActivationFunction.SIGMOID, neuronsPerLayer);
    }

    public NeuralNet(CostFunction costFunction, ActivationFunction hiddenActivationFunction, ActivationFunction outputActivationFunction, int ... neuronsPerLayer) {
        this.costFunction = costFunction;
        this.activationFunction = hiddenActivationFunction;
        this.outputActivationFunction = outputActivationFunction;
        numberOfLayers = neuronsPerLayer.length;

        weightMatrixes = new Matrix[numberOfLayers-1];
        biasVectors = new Vector[numberOfLayers-1];

        for(int i=0;i<numberOfLayers-1;i++) {
            int numberOfNeuronsInCurrentLayer = neuronsPerLayer[i];
            int numberOfNeuronsInNextLayer = neuronsPerLayer[i+1];

            weightMatrixes[i] = new Matrix(numberOfNeuronsInNextLayer, numberOfNeuronsInCurrentLayer);
            biasVectors[i] = new Vector(numberOfNeuronsInNextLayer);
        }

        randomizeWeights();
    }

    private void randomizeWeights() {
        Random random = new Random();
        for(Matrix matrix : weightMatrixes) {
            // He initialization — optimal for ReLU: σ = √(2 / nrInputs)
            double sigma = MLMath.sqrt(2.0 / matrix.getNrColumns());
            matrix.apply(x -> random.nextGaussian() * sigma);
        }

        for(Vector biasVector : biasVectors) {
            biasVector.apply(i -> random.nextDouble(-0.01, 0.01));
        }
    }

    public void learn(List<DataPoint> dataPoints) {
        for(int epoch = 0; epoch < numberOfEpochs; epoch++) {
            double lr = learningRateAt(epoch);
            Collections.shuffle(dataPoints);
            for(DataPoint dataPoint : dataPoints) {
                Vector[] activations = forwardPass(dataPoint.input());
                Gradient gradient = calculateGradient(activations, dataPoint.output());
                for(int i = 0; i < numberOfLayers - 1; i++) {
                    weightMatrixes[i] = weightMatrixes[i].minus(gradient.dWs()[i].scale(lr));
                    biasVectors[i] = biasVectors[i].minus(gradient.dbs()[i].scale(lr));
                }
            }
            logger.log("Cost at epoch " + epoch + " (lr=" + lr + "): " + cost(dataPoints));
        }
    }

    // step decay: halve the learning rate every lrDecaySteps epochs
    double learningRateAt(int epoch) {
        if(lrDecaySteps == 0) return learningRate;
        int steps = epoch / lrDecaySteps;
        return learningRate * Math.pow(lrDecayFactor, steps);
    }

    private Vector[] forwardPass(Vector input) {
        Vector[] activations = new Vector[numberOfLayers];
        activations[0] = input;
        for(int i = 0; i < numberOfLayers - 1; i++) {
            ActivationFunction af = (i == numberOfLayers - 2) ? outputActivationFunction : activationFunction;
            activations[i+1] = af.applyVector(weightMatrixes[i].multiply(activations[i]).plus(biasVectors[i]));
        }
        return activations;
    }

    private Gradient calculateGradient(Vector[] activations, Vector target) {
        int n = weightMatrixes.length;
        Matrix[] dWs = new Matrix[n];
        Vector[] dbs = new Vector[n];
        dbs[n-1] = outputActivationFunction.outputDelta(activations[n], target, costFunction);
        dWs[n-1] = dbs[n-1].multiply(activations[n-1]);

        for (int i = n-2; i >= 0; i--) {
            Vector wTDelta = weightMatrixes[i+1].transpose().multiply(dbs[i+1]);
            Vector activationGrad = activations[i+1].map(activationFunction::derivative);
            dbs[i] = wTDelta.hadamardProduct(activationGrad);
            dWs[i] = dbs[i].multiply(activations[i]);
        }

        return new Gradient(dWs, dbs);
    }

    public Vector predict(Vector input) {
        Vector vector = input;
        for(int i = 0; i < numberOfLayers - 1; i++) {
            ActivationFunction af = (i == numberOfLayers - 2) ? outputActivationFunction : activationFunction;
            vector = af.applyVector(weightMatrixes[i].multiply(vector).plus(biasVectors[i]));
        }
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
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setLearningRateDecay(int steps, double factor) {
        this.lrDecaySteps = steps;
        this.lrDecayFactor = factor;
    }

    private static record Gradient(Matrix[] dWs, Vector[] dbs) {}

}
