package kits.ml.core;

import java.util.stream.Collectors;
import java.util.stream.Stream;

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
		return Stream.of(values).map(String::valueOf).collect(Collectors.joining(" "));
	}
}
