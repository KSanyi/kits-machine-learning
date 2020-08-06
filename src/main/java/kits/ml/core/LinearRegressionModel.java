package kits.ml.core;

import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import Jama.Matrix;
import kits.ml.core.math.MLMath;
import kits.ml.core.math.Statistics;
import kits.ml.core.math.Statistics.Standardizer;

public class LinearRegressionModel implements MLModel {

    private final int steps;
    private final double alpha;

    private final int inputDimension;

    private double[] parameters;

    private Standardizer[] standardizers;

    public LinearRegressionModel(int inputDimension) {
        this(inputDimension, 0.01);
    }

    public LinearRegressionModel(int inputDimension, double alpha) {
        this(inputDimension, alpha, 1000);
    }

    public LinearRegressionModel(int inputDimension, double alpha, int steps) {
        this.inputDimension = inputDimension;
        this.alpha = alpha;
        this.steps = steps;
        parameters = new double[inputDimension + 1];
    }

    public void setParameters(double ... parameters) {
        if (parameters.length != inputDimension + 1) {
            throw new IllegalArgumentException("Parameters must have dimension " + (inputDimension + 1));
        }
        this.parameters = parameters;
        this.standardizers = IntStream.range(0, parameters.length - 1).mapToObj(i -> Statistics.NoOpStandardizer).toArray(Standardizer[]::new);
    }

    @Override
    public void learn(List<LearningData> learningDataSet) {
        learningDataSet.stream().map(learningData -> learningData.input).forEach(this::checkDimension);

        standardizers = createStandardizers(learningDataSet);

        Matrix X = getStandardizedInputMatrix(learningDataSet, standardizers);
        Matrix y = getOutputVector(learningDataSet);
        Matrix theta = new Matrix(parameters, parameters.length);

        double prevCost = 100;
        for(int i=0;i<steps;i++) {
            parameters = theta.getColumnPackedCopy();
            System.out.println("Params: " + Arrays.toString(parameters));
            double cost = calculateCost(learningDataSet);
            System.out.println("Cost: " + cost);
            System.out.println("Cost decrement: " + (prevCost - cost));
            prevCost = cost;
            
            /**
             * theta - alpha / n * X' * (X * theta - y)
             */
            theta = theta.minus(X.transpose().times(X.times(theta).minus(y)).times(alpha / learningDataSet.size()));
        }

        parameters = theta.getColumnPackedCopy();
    }

    private Standardizer[] createStandardizers(List<LearningData> learningDataSet) {
        return IntStream.range(0, inputDimension)
                .mapToObj(i -> new Standardizer(getColumnValues(learningDataSet, i)))
                .toArray(Standardizer[]::new);
    }

    private static double[] getColumnValues(List<LearningData> learningDataSet, int i) {
        return learningDataSet.stream()
                .mapToDouble(learningData -> learningData.input.values[i])
                .toArray();
    }

    private Matrix getStandardizedInputMatrix(List<LearningData> learningDataSet, Standardizer[] standardizers) {
        double[] values = learningDataSet.stream()
                .flatMapToDouble(learningData -> DoubleStream.concat(DoubleStream.of(1), DoubleStream.of(Statistics.standardize(learningData.input.values, standardizers))))
                .toArray();
        return new Matrix(values, inputDimension + 1).transpose();
    }

    private static Matrix getOutputVector(List<LearningData> learningDataSet) {
        double[] values = learningDataSet.stream()
                .mapToDouble(learningData -> learningData.output)
                .toArray();
        return new Matrix(values, learningDataSet.size());
    }

    @Override
    public double calculateOutput(Input input) {
        checkDimension(input);
        return parameters[0] + IntStream.range(0, inputDimension).mapToDouble(i -> parameters[i + 1] * standardizers[i].standardize(input.values[i])).sum();
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
