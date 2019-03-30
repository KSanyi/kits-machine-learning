package kits.ml.core;

import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import Jama.Matrix;
import kits.ml.core.math.MLMath;

public class SimpleLinearRegressionModel implements MLModel {

    private final int inputDimension;

    public SimpleLinearRegressionModel(int inputDimension) {
        this.inputDimension = inputDimension;
        this.parameters = new double[inputDimension + 1];
    }

    private double[] parameters;

    @Override
    public void learn(List<LearningData> learningDataSet) {
        learningDataSet.stream().map(learningData -> learningData.input).forEach(this::checkDimension);

        Matrix X = getInputMatrix(learningDataSet);
        Matrix y = getOutputVector(learningDataSet);

        /**
         * inv(X' * X) * X' * y
         */
        Matrix theta = X.transpose().times(X).inverse().times(X.transpose()).times(y);
        parameters = theta.getColumnPackedCopy();
    }

    private Matrix getInputMatrix(List<LearningData> learningDataSet) {
        double[] values = learningDataSet.stream().flatMapToDouble(learningData -> DoubleStream.concat(DoubleStream.of(1), DoubleStream.of(learningData.input.values))).toArray();
        return new Matrix(values, inputDimension + 1).transpose();
    }

    private Matrix getOutputVector(List<LearningData> learningDataSet) {
        double[] values = learningDataSet.stream().mapToDouble(learningData -> learningData.output).toArray();
        return new Matrix(values, learningDataSet.size());
    }

    @Override
    public double calculateOutput(Input input) {
        checkDimension(input);
        return parameters[0] + IntStream.range(0, inputDimension).mapToDouble(i -> parameters[i + 1] * input.values[i]).sum();
    }

    @Override
    public double calculateCost(List<LearningData> learningDataSet) {
        int n = learningDataSet.size();
        return learningDataSet.stream().mapToDouble(learningData -> MLMath.square(learningData.output - calculateOutput(learningData.input))).sum() / (2 * n);
    }

    private void checkDimension(Input input) {
        if (input.dimension() != inputDimension)
            throw new IllegalArgumentException("Input dimension must be " + inputDimension);
    }

}