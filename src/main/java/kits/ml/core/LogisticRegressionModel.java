package kits.ml.core;

import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import kits.ml.core.math.MLMath;
import kits.ml.core.math.linalg.Matrix;
import kits.ml.core.math.linalg.Vector;

public class LogisticRegressionModel {

    private final int steps;

    private final int inputDimension;

    Vector parameters;

    private final double lambda;

    public LogisticRegressionModel(int inputDimension) {
        this(inputDimension, 1000, 0);
    }

    public LogisticRegressionModel(int inputDimension, int steps, double lambda) {
        this.inputDimension = inputDimension;
        this.steps = steps;
        this.lambda = lambda;
        parameters = new Vector(inputDimension + 1);
    }

    public void setParameters(double... parameters) {
        if (parameters.length != inputDimension + 1) {
            throw new IllegalArgumentException("Parameters must have dimension " + (inputDimension + 1));
        }
        this.parameters = new Vector(parameters);
    }

    public void learn(List<LearningData> learningDataSet) {

        if (learningDataSet.isEmpty())
            throw new IllegalArgumentException("Can not learn from empty data set");

        learningDataSet.stream().map(learningData -> learningData.input).forEach(this::checkDimension);

        Matrix X = getInputMatrix(learningDataSet);
        Vector y = getOutputVector(learningDataSet);
        Vector theta = parameters;

        for (int i = 0; i < steps; i++) {
            parameters = theta;
            // System.out.println("Params: " + Arrays.toString(parameters));
            // System.out.println("Cost: " + calculateCost(learningDataSet));

            /**
             * theta - alpha / n * X' * (sigmoid(X * theta) - y)
             */
            Vector gradient = X.transpose().multiply(MLMath.sigmoid(X.multiply(theta)).minus(y)).multiply(1d / learningDataSet.size());
            double alpha = findAlpha(learningDataSet, gradient, theta);
            theta = theta.minus(gradient.multiply(alpha));
        }

        parameters = theta;

        for (int i=0;i<parameters.length();i++) {
            System.out.format("%.5f ", parameters.get(i));
        }
    }

    private static double findAlpha(List<LearningData> learningDataSet, Vector gradient, Vector currentTheta) {
        return 0.0001;

        /*
         * double currentCost = calculateCost(learningDataSet);
         * 
         * Matrix theta = currentTheta; double minCost = currentCost; double bestAlpha =
         * Math.pow(10, 5); for(int i=0;i<10;i++) { double alphaCandidate = 1d /
         * Math.pow(10, i); theta = currentTheta.minus(gradient.times(alphaCandidate));
         * parameters = theta.getColumnPackedCopy(); double cost =
         * calculateCost(learningDataSet); //System.out.println("Cost for " +
         * alphaCandidate + ": " + cost); if(cost < minCost) { minCost = cost; bestAlpha
         * = alphaCandidate; } }
         * 
         * return bestAlpha2;
         */
    }

    private static Matrix getInputMatrix(List<LearningData> learningDataSet) {
        
        double[][] values = learningDataSet.stream()
                .map(learningData -> DoubleStream.concat(DoubleStream.of(1), DoubleStream.of(learningData.input.values)).toArray())
                .toArray(double[][]::new);
        return new Matrix(values);
    }

    private static Vector getOutputVector(List<LearningData> learningDataSet) {
        double[] values = learningDataSet.stream().mapToDouble(learningData -> learningData.output).toArray();
        return new Vector(values);
    }

    public double calculateOutput(Input input) {
        checkDimension(input);
        return MLMath.sigmoid(parameters.get(0) + IntStream.range(0, inputDimension).mapToDouble(i -> parameters.get(i + 1) * input.values[i]).sum());
    }

    public int predict(Input input) {
        double output = calculateOutput(input);
        return output > 0.5 ? 1 : 0;
    }

    public double calculateCost(List<LearningData> learningDataSet) {
        int n = learningDataSet.size();

        double cost = learningDataSet.stream().mapToDouble(this::calculateCost).sum() / n;
        DoubleStream paramsToRegularize = IntStream.range(1, parameters.length()).mapToDouble(i -> parameters.get(i));
        double regularizedCost = lambda * paramsToRegularize.map(MLMath::square).sum() / (2 * n);
        return cost + regularizedCost;
    }

    private double calculateCost(LearningData learningData) {
        Input input = learningData.input;
        double output = learningData.output;
        double calculatedOutput = calculateOutput(input);

        return -output * Math.log(calculatedOutput) - (1 - output) * Math.log(1 - calculatedOutput);
    }

    public Vector calculateGradient(List<LearningData> learningDataSet) {
        Matrix X = getInputMatrix(learningDataSet);
        Vector y = getOutputVector(learningDataSet);

        Vector theta = new Vector(parameters);
        Vector thetaForRegularization = new Vector(parameters);
        thetaForRegularization.set(0, 0);
        int n = learningDataSet.size();

        return X.transpose().multiply(MLMath.sigmoid(X.multiply(theta)).minus(y)).multiply(1d / n).plus(thetaForRegularization.multiply(lambda / n));
    }

    private void checkDimension(Input input) {
        if (input.dimension() != inputDimension) {
            throw new IllegalArgumentException("Input dimension must be " + inputDimension);
        }
    }

}
