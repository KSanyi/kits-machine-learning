package kits.ml.core.math;

import java.util.stream.DoubleStream;

import Jama.Matrix;

public class MLMath {

    public static double square(double x) {
        return x * x;
    }

    public static double sqrt(double x) {
        return Math.sqrt(x);
    }

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double[] sigmoid(double[] values) {
        return DoubleStream.of(values).map(MLMath::sigmoid).toArray();
    }

    public static Matrix sigmoid(Matrix X) {
        double[] sigmoidValues = sigmoid(X.getColumnPackedCopy());
        return new Matrix(sigmoidValues, X.getRowDimension());
    }

    public static double sigmoidGradient(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    public static double[] sigmoidGradient(double[] values) {
        return DoubleStream.of(values).map(MLMath::sigmoidGradient).toArray();
    }

    public static Matrix sigmoidGradient(Matrix X) {
        double[] sigmoidGradientValues = sigmoidGradient(X.getColumnPackedCopy());
        return new Matrix(sigmoidGradientValues, X.getRowDimension());
    }

}
