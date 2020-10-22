package kits.ml.core;

import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import kits.ml.core.math.MLMath;
import kits.ml.core.math.MLStat;
import kits.ml.core.math.MLStat.Standardizer;
import kits.ml.core.math.linalg.Matrix;
import kits.ml.core.math.linalg.Vector;

public class LinearRegressionModel implements MLModel {

    private final int steps;
    private final double alpha;

    private final int inputDimension;

    private Vector parameters;

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
        parameters = new Vector(inputDimension + 1);
    }

    public void setParameters(double ... parameters) {
        if (parameters.length != inputDimension + 1) {
            throw new IllegalArgumentException("Parameters must have dimension " + (inputDimension + 1));
        }
        this.parameters = new Vector(parameters);
        this.standardizers = IntStream.range(0, parameters.length - 1).mapToObj(i -> MLStat.NoOpStandardizer).toArray(Standardizer[]::new);
    }

    @Override
    public void learn(List<LearningData> learningDataSet) {
        learningDataSet.stream().map(learningData -> learningData.input).forEach(this::checkDimension);

        standardizers = createStandardizers(learningDataSet);

        Matrix X = getStandardizedInputMatrix(learningDataSet, standardizers);
        Vector y = getOutputVector(learningDataSet);
        Vector theta = parameters;

        double prevCost = 100;
        for(int i=0;i<steps;i++) {
            System.out.println("Params: " + theta);
            double cost = calculateCost(learningDataSet);
            System.out.println("Cost: " + cost);
            System.out.println("Cost decrement: " + (prevCost - cost));
            prevCost = cost;
            
            /**
             * theta - alpha / n * X' * (X * theta - y)
             */
            theta = theta.minus(X.transpose().multiply(X.multiply(theta).minus(y)).multiply(alpha / learningDataSet.size()));
        }

        parameters = theta;
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

    private static Matrix getStandardizedInputMatrix(List<LearningData> learningDataSet, Standardizer[] standardizers) {
        double[][] values = learningDataSet.stream()
                .map(learningData -> DoubleStream.concat(DoubleStream.of(1), DoubleStream.of(MLStat.standardize(learningData.input.values, standardizers))).toArray())
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
        return parameters.get(0) + IntStream.range(0, inputDimension).mapToDouble(i -> parameters.get(i + 1) * standardizers[i].standardize(input.values[i])).sum();
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
