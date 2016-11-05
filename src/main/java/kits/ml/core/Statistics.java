package kits.ml.core;

import java.util.stream.DoubleStream;

public class Statistics {

	public static double average(double[] values) {
		return DoubleStream.of(values).sum() / values.length;
	}
	
	public static double stDev(double[] values) {
		double average = average(values);
		return Math.sqrt(DoubleStream.of(values).map(value -> Math.square(value - average)).sum() / values.length);
	}
	
}
