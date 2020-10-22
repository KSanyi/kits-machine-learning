package kits.ml.core.math;

import java.util.ArrayList;
import java.util.List;

import kits.ml.core.math.linalg.Matrix;
import kits.ml.core.math.linalg.Vector;

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

    public static Matrix sigmoid(Matrix X) {
        return X.map((i, j) -> MLMath.sigmoid(X.get(i, j)));
    }
    
    public static Vector sigmoid(Vector x) {
        return x.map((i -> MLMath.sigmoid(x.get(i))));
    }

    public static double sigmoidGradient(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    public static Matrix sigmoidGradient(Matrix X) {
        return X.map((i, j) -> MLMath.sigmoidGradient(X.get(i, j)));
    }
    
    public static double[] generateArithmeticSeries(double start, double diff, double end) {
        
        List<Double> values = new ArrayList<>();
        double value = start;
        while(value < end) {
            values.add(value);
            value += diff;
        }
        values.add(end);
        
        return values.stream().mapToDouble(d -> d).toArray();
    }

}
