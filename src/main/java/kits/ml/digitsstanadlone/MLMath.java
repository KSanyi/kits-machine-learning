package kits.ml.digitsstanadlone;

public class MLMath {

    public static double sqrt(double x) {
        return Math.sqrt(x);
    }

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static Vector sigmoid(Vector x) {
        return x.map((i -> MLMath.sigmoid(x.get(i))));
    }

}
