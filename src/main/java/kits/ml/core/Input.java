package kits.ml.core;

import java.util.Arrays;

public class Input {

	final double[] values;

	public Input(double ... values){
		this.values = values;
	}
	
	public int dimension() {
		return values.length;
	}
	
	@Override
	public String toString() {
		return Arrays.toString(values);
	}
}
