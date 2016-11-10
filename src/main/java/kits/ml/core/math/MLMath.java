package kits.ml.core.math;

import java.util.stream.DoubleStream;

public class MLMath {

	public static double square(double value) {
		return value * value;
	}
	
	public static double sqrt(double value) {
		return java.lang.Math.sqrt(value);
	}
	
	public static double sigmoid(double value) {
		return 1 / (1 + java.lang.Math.exp(-value));
	}
	
	public static double[] sigmoid(double[] values) {
	    return DoubleStream.of(values).map(MLMath::sigmoid).toArray();
    }
	
}
