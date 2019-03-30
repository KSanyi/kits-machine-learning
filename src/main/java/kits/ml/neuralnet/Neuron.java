package kits.ml.neuralnet;

import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import kits.ml.core.Input;
import kits.ml.core.math.MLMath;

public class Neuron {

    private final double[] weights;

    private final int inputDimension;

    public Neuron(int inputDimension) {
        this.inputDimension = inputDimension;
        weights = new double[inputDimension + 1];
    }

    public Neuron(double... weights) {
        this.weights = weights;
        inputDimension = weights.length - 1;
    }

    public double calculateOutput(Input input) {
        return MLMath.sigmoid(weights[0] + IntStream.range(0, inputDimension).mapToDouble(i -> weights[i + 1] * input.values[i]).sum());
    }

    public DoubleStream weightsToRegularize() {
        return IntStream.range(1, weights.length).mapToDouble(i -> weights[i]);
    }

}
