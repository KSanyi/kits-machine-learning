package kits.ml.neuralnet;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import kits.ml.core.Input;
import kits.ml.core.LearningData;
import kits.ml.core.math.MLMath;

public class NeuralNet {

    private final List<Layer> hiddenLayers;

    private final double lambda;

    public NeuralNet(List<double[][]> layerWeights, double lambda) {
        hiddenLayers = new ArrayList<>();
        for (double[][] weights : layerWeights) {
            hiddenLayers.add(new Layer(weights));
        }
        this.lambda = lambda;
    }

    public NeuralNet(int inputDimension, double lambda, int... hiddenLayerNeurons) {
        if (hiddenLayerNeurons.length == 0)
            throw new IllegalArgumentException("At least 1 layer is expected");

        hiddenLayers = new ArrayList<>();
        int numberOfNeuronsInPrevLayer = inputDimension;
        for (int numberOfNeurons : hiddenLayerNeurons) {
            hiddenLayers.add(new Layer(numberOfNeurons, numberOfNeuronsInPrevLayer));
            numberOfNeuronsInPrevLayer = numberOfNeurons;
        }

        this.lambda = lambda;
    }

    public int predict(Input input) {
        return findIndexForMaxOutput(calculateOutput(input)) + 1;
    }

    private static int findIndexForMaxOutput(double[] output) {
        double max = 0;
        int indexForMax = -1;
        for (int i = 0; i < output.length; i++) {
            // System.out.println("Output for " + (i+1) + " " + output[i]);
            if (output[i] > max) {
                max = output[i];
                indexForMax = i;
            }
        }
        return indexForMax;
    }

    public double calculateCost(List<LearningData> learningDataSet) {
        int n = learningDataSet.size();

        double cost = learningDataSet.stream().mapToDouble(this::calculateCost).sum() / n;
        double regularizedCost = lambda * weightsToRegularize().map(MLMath::square).sum() / (2 * n);
        return cost + regularizedCost;
    }

    private DoubleStream weightsToRegularize() {
        return hiddenLayers.stream().flatMapToDouble(Layer::weightsToRegularize);
    }

    private double calculateCost(LearningData learningData) {
        Input input = learningData.input();
        double[] calculatedOutput = calculateOutput(input);
        double[] expectedOutput = calculateExpectedOutputArray(learningData.output(), calculatedOutput.length);

        return IntStream.range(0, calculatedOutput.length)
                .mapToDouble(i -> -expectedOutput[i] * Math.log(calculatedOutput[i]) - (1 - expectedOutput[i]) * Math.log(1 - calculatedOutput[i])).sum();
    }

    private static double[] calculateExpectedOutputArray(double output, int size) {
        int outputIndex = (int) output - 1;
        double[] expectedOutput = new double[size];
        expectedOutput[outputIndex] = 1;
        return expectedOutput;
    }

    public double[] calculateOutput(Input input) {
        Input inp = input;
        double[] output = null;
        for (Layer hiddenLayer : hiddenLayers) {
            output = hiddenLayer.calculateOutput(inp);
            inp = new Input(output);
        }

        return output;
    }

    private static class Layer {

        final List<Neuron> neurons;

        Layer(int numberOfNeurons, int numberOfNeuronsInPrevLayer) {
            neurons = IntStream.range(0, numberOfNeurons).mapToObj(i -> new Neuron(numberOfNeuronsInPrevLayer)).collect(Collectors.toList());
        }

        Layer(double[][] weights) {
            neurons = IntStream.range(0, weights.length).mapToObj(i -> new Neuron(weights[i])).collect(Collectors.toList());
        }

        double[] calculateOutput(Input input) {
            return neurons.stream().mapToDouble(neuron -> neuron.calculateOutput(input)).toArray();
        }

        DoubleStream weightsToRegularize() {
            return neurons.stream().flatMapToDouble(Neuron::weightsToRegularize);
        }
    }

}
