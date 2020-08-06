package kits.ml.core.math;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

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
    
    public static double[] generate(double start, double diff, double end) {
        
        List<Double> values = new ArrayList<>();
        double value = start;
        while(value < end) {
            values.add(value);
            value += diff;
        }
        values.add(end);
        
        return values.stream().mapToDouble(d -> d).toArray();
    }
    
    public static Matrix matrixFromColumns(double[] ... cols) {
        
        if(cols.length == 0) throw new IllegalArgumentException("At least 1 column is required");
        
        int rowCount = cols[0].length;
        
        if(Stream.of(cols).anyMatch(c -> c.length != rowCount)) {
            throw new IllegalArgumentException("Columns must have the same size");
        }
        
        double[] rowPackedValues = new double[cols.length * rowCount];
        int k = 0;
        for(int i=0;i<cols.length;i++) {
            for(int j=0;j<cols[0].length;j++) {
                rowPackedValues[k++] = cols[i][j];
            }
        }
        
        return new Matrix(rowPackedValues, rowCount);
    }

}
