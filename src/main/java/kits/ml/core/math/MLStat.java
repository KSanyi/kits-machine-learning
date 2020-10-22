package kits.ml.core.math;

import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class MLStat {

    public static double average(double[] values) {
        if (values.length == 0)
            throw new IllegalArgumentException("Cannot compute statistics without data");
        return DoubleStream.of(values).average().getAsDouble();
    }

    public static double stDev(double[] values) {
        double average = average(values);
        return MLMath.sqrt(DoubleStream.of(values).map(value -> MLMath.square(value - average)).average().getAsDouble());
    }

    public static double[] standardize(double[] values, Standardizer[] standardizers) {
        if (values.length != standardizers.length)
            throw new IllegalArgumentException();

        return IntStream.range(0, values.length).mapToDouble(i -> standardizers[i].standardize(values[i])).toArray();
    }

    public static record Standardizer(double average, double stdev) {

        public Standardizer(double[] values) {
            average = MLStat.average(values);
            stdev = MLStat.stDev(values);
        }

        public double standardize(double value) {
            return (value - average) / stdev;
        }

        Standardizer() {
            average = 0;
            stdev = 1;
        }
    }

    public static Standardizer NoOpStandardizer = new Standardizer();

}
