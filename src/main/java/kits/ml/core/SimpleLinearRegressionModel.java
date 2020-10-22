package kits.ml.core;

import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import kits.ml.core.math.MLMath;
import kits.ml.core.math.linalg.GaussEliminationCalculator;
import kits.ml.core.math.linalg.Matrix;
import kits.ml.core.math.linalg.Vector;

public class SimpleLinearRegressionModel implements MLModel {

    private final int inputDimension;
    
    private Vector parameters;

    public SimpleLinearRegressionModel(int inputDimension) {
        this.inputDimension = inputDimension;
        this.parameters = new Vector(inputDimension + 1);
    }

    @Override
    public void learn(List<LearningData> learningDataSet) {
        learningDataSet.stream().map(learningData -> learningData.input).forEach(this::checkDimension);

        Matrix X = getInputMatrix(learningDataSet);
        Vector y = getOutputVector(learningDataSet);

        /**
         * inv(X' * X) * X' * y
         */
        Vector theta = inverse(X.transpose().multiply(X)).multiply(X.transpose()).multiply(y);
        parameters = theta;
    }
    
    private static Matrix inverse(Matrix A) {
        return GaussEliminationCalculator.calculateInverse(A);
    }

    private static Matrix getInputMatrix(List<LearningData> learningDataSet) {
        double[][] values = learningDataSet.stream()
                .map(learningData -> DoubleStream.concat(DoubleStream.of(1), DoubleStream.of(learningData.input.values)).toArray())
                .toArray(double[][]::new);
        return new Matrix(values);
    }

    private static Vector getOutputVector(List<LearningData> learningDataSet) {
        double[] values = learningDataSet.stream()
                .mapToDouble(learningData -> learningData.output)
                .toArray();
        return new Vector(values);
    }

    @Override
    public double calculateOutput(Input input) {
        checkDimension(input);
        return parameters.get(0) + IntStream.range(0, inputDimension).mapToDouble(i -> parameters.get(i + 1) * input.values[i]).sum();
    }

    @Override
    public double calculateCost(List<LearningData> learningDataSet) {
        int n = learningDataSet.size();
        return learningDataSet.stream()
                .mapToDouble(learningData -> MLMath.square(learningData.output - calculateOutput(learningData.input)))
                .sum() / (2 * n);
    }

    private void checkDimension(Input input) {
        if (input.dimension() != inputDimension)
            throw new IllegalArgumentException("Input dimension must be " + inputDimension);
    }

}