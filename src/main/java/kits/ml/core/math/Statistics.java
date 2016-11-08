package kits.ml.core.math;

import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class Statistics {

	public static double average(double[] values) {
		if(values.length == 0) throw new IllegalArgumentException("Cannot compute statistics without data");
		return DoubleStream.of(values).sum() / values.length;
	}
	
	public static double stDev(double[] values) {
		double average = average(values);
		return MLMath.sqrt(DoubleStream.of(values).map(value -> MLMath.square(value - average)).sum() / values.length);
	}
	
	public static double[] standardize(double[] values, Standardizer[] standardizers) {
		if(values.length != standardizers.length) throw new IllegalArgumentException();
		
		return IntStream.range(0, values.length).mapToDouble(i -> standardizers[i].standardize(values[i])).toArray();
	}
	
	public static class Standardizer {
		
		private final double average;
		private final double stdev;

		public Standardizer(double[] values) {
			average = Statistics.average(values);
			stdev = Statistics.stDev(values);
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
