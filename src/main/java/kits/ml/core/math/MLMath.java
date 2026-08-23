package kits.ml.core.math;

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
        return X.map(MLMath::sigmoid);
    }
    
    public static Vector sigmoid(Vector x) {
        return x.map(MLMath::sigmoid);
    }

    public static double sigmoidGradient(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    public static Matrix sigmoidGradient(Matrix X) {
        return X.map(MLMath::sigmoidGradient);
    }
    
    public static Vector sigmoidGradient(Vector x) {
        return x.map(MLMath::sigmoidGradient);
    }

    public static Vector oneHot(int length, int index) {
        Vector result = new Vector(length);
        result.set(index, 1);
        return result;
    }

    public static Vector softMax(Vector vector) {
        
        double sum = 0;
        Vector result = new Vector(vector.length());
        for(int i=0;i<vector.length();i++) {
            result.set(i, Math.exp(-vector.get(i))); 
            sum += result.get(i); 
        }
        result.scaleThis(1/sum);
        
        return result;
    }
    
}
