package kits.ml.regression;

import java.util.List;
import java.util.function.Function;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import kits.ml.core.LearningData;
import kits.ml.core.math.MLMath;
import kits.ml.core.math.MLStat;
import kits.ml.core.math.MLStat.Standardizer;
import kits.ml.core.math.linalg.Matrix;
import kits.ml.core.math.linalg.Vector;
import kits.ml.core.math.optimization.GradientDescentOptimizer;
import kits.ml.core.math.optimization.GradientOptimizer;

public class LinearRegressionModel implements MLModel {

    private final int inputDimension;

    private Vector parameters;

    private Standardizer[] standardizers;
    
    private final GradientOptimizer optimizer;

    public LinearRegressionModel(int inputDimension) {
        this(inputDimension, new GradientDescentOptimizer(100, 0.01, 0.001));
    }

    public LinearRegressionModel(int inputDimension, GradientOptimizer optimizer) {
        this.inputDimension = inputDimension;
        this.optimizer = optimizer;
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
        learningDataSet.stream().map(learningData -> learningData.input()).forEach(this::checkDimension);

        standardizers = createStandardizers(learningDataSet);

        Matrix X = createStandardizedInputMatrix(learningDataSet, standardizers);
        Vector y = createOutputVector(learningDataSet);
        Vector theta = parameters;
        
        Function<Vector, Double> costFunction = v -> calculateCost(learningDataSet, v);
        // X' * (X * theta - y)
        parameters = optimizer.optimize(theta, costFunction, v -> X.transpose().multiply(X.multiply(v).minus(y)));

//        double prevCost = 100;
//        for(int i=0;i<steps;i++) {
//            System.out.println("Params: " + theta);
//            double cost = calculateCost(learningDataSet);
//            System.out.println("Cost: " + prevCost + " -> " + cost + "(" + (cost - prevCost) + ")");
//            prevCost = cost;
//            
//            /**
//             * theta - alpha / n * X' * (X * theta - y)
//             */
//            theta = theta.minus(X.transpose().multiply(X.multiply(theta).minus(y)).scale(alpha / learningDataSet.size()));
//            parameters = theta;
//        }

    }

    private Standardizer[] createStandardizers(List<LearningData> learningDataSet) {
        return IntStream.range(0, inputDimension)
                .mapToObj(i -> new Standardizer(getColumnValues(learningDataSet, i)))
                .toArray(Standardizer[]::new);
    }

    private static double[] getColumnValues(List<LearningData> learningDataSet, int i) {
        return learningDataSet.stream()
                .mapToDouble(learningData -> learningData.input().get(i))
                .toArray();
    }

    private static Matrix createStandardizedInputMatrix(List<LearningData> learningDataSet, Standardizer[] standardizers) {
        double[][] values = learningDataSet.stream()
                .map(learningData -> DoubleStream.concat(DoubleStream.of(1), DoubleStream.of(MLStat.standardize(learningData.input(), standardizers))).toArray())
                .toArray(double[][]::new);
        return new Matrix(values);
    }

    private static Vector createOutputVector(List<LearningData> learningDataSet) {
        double[] values = learningDataSet.stream()
                .mapToDouble(learningData -> learningData.output())
                .toArray();
        return new Vector(values);
    }

    @Override
    public double calculateOutput(Vector input) {
        checkDimension(input);
        return parameters.get(0) + IntStream.range(0, inputDimension).mapToDouble(i -> parameters.get(i + 1) * standardizers[i].standardize(input.get(i))).sum();
    }

    @Override
    public double calculateCost(List<LearningData> learningDataSet) {
        int n = learningDataSet.size();
        return learningDataSet.stream().mapToDouble(learningData -> MLMath.square(learningData.output() - calculateOutput(learningData.input()))).sum() / (2 * n);
    }
    
    private double calculateCost(List<LearningData> learningDataSet, Vector weights) {
        int n = learningDataSet.size();
        return learningDataSet.stream().mapToDouble(learningData -> MLMath.square(learningData.output() - calculateOutput(learningData.input(), weights))).sum() / (2 * n);
    }
    
    public double calculateOutput(Vector input, Vector weights) {
        checkDimension(input);
        return weights.get(0) + IntStream.range(0, inputDimension).mapToDouble(i -> weights.get(i + 1) * standardizers[i].standardize(input.get(i))).sum();
    }

    private void checkDimension(Vector input) {
        if (input.length() != inputDimension)
            throw new IllegalArgumentException("Input dimension must be " + inputDimension);
    }

}
