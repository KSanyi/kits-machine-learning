package kits.ml.core;

import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import Jama.Matrix;

public class LinearRegressionModel implements MLModel {

	private final int STEPS = 1000;
	private final double learningRate = 0.005;
	
	private final int inputDimension;
	
	public LinearRegressionModel(int inputDimension) {
		this.inputDimension = inputDimension;
		parameters = new double[inputDimension+1];
	}

	private double[] parameters;
	
	@Override
	public void learn(List<LearningData> learningDataSet) {
		learningDataSet.stream().map(learningData -> learningData.input).forEach(this::checkDimension);

		Matrix X = convertToMatrix(learningDataSet);
		Matrix y = convertToVector(learningDataSet);
		Matrix theta = new Matrix(parameters, parameters.length);
		
		for(int i=0;i<STEPS;i++) {
			theta = theta.minus(X.transpose().times(X.times(theta).minus(y)).times(learningRate / inputDimension));
		}
		
		parameters = theta.getColumnPackedCopy();
	}
	
	private Matrix convertToMatrix(List<LearningData> learningDataSet) {
		double[] values = learningDataSet.stream().flatMapToDouble(learningData -> DoubleStream.concat(DoubleStream.of(1), DoubleStream.of(learningData.input.values))).toArray();
		return new Matrix(values, inputDimension+1).transpose();
	}
	
	private Matrix convertToVector(List<LearningData> learningDataSet) {
		double[] values = learningDataSet.stream().mapToDouble(learningData -> learningData.output).toArray();
		return new Matrix(values, learningDataSet.size());
	}

	@Override
	public double calculateOutput(Input input) {
		checkDimension(input);
		return parameters[0] + IntStream.range(0, inputDimension).mapToDouble(i -> parameters[i+1] * input.values[i]).sum();
	}

	@Override
	public double calculateCost(List<LearningData> learningDataSet) {
		int n = learningDataSet.size();
		return learningDataSet.stream().mapToDouble(learningData -> square(learningData.output - calculateOutput(learningData.input))).sum() / (2 * n);
	}
	
	private double square(double d) {
		return d * d;
	}
	
	private void checkDimension(Input input) {
		if(input.dimension() != inputDimension) throw new IllegalArgumentException("Input dimension must be " + inputDimension);
	}

}
