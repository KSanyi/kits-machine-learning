package kits.ml.core;

import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import Jama.Matrix;

public class LinearRegressionModel implements MLModel {

	private final int steps;
	private final double alpha;
	
	private final int inputDimension;
	
	public LinearRegressionModel(int inputDimension) {
		this(inputDimension, 0.1);
	}
	
	public LinearRegressionModel(int inputDimension, double alpha) {
		this(inputDimension, alpha, 1000);
	}
	
	public LinearRegressionModel(int inputDimension, double alpha, int steps) {
		this.inputDimension = inputDimension;
		this.alpha = alpha;
		this.steps = steps;
		parameters = new double[inputDimension+1];
	}

	private double[] parameters;
	
	private Normalizer[] normalizers;
	
	@Override
	public void learn(List<LearningData> learningDataSet) {
		learningDataSet.stream().map(learningData -> learningData.input).forEach(this::checkDimension);

		normalizers = createNormalizers(learningDataSet);
		
		Matrix X = convertToNormalizedMatrix(learningDataSet, normalizers);
		Matrix y = convertToVector(learningDataSet);
		Matrix theta = new Matrix(parameters, parameters.length);
		
		for(int i=0;i<steps;i++) {
			parameters = theta.getColumnPackedCopy();
			System.out.println("Params: " + Arrays.toString(parameters));
			System.out.println("Cost: " + calculateCost(learningDataSet));
			// theta - alpha / n * X' * (X * theta - y)
			
			theta = theta.minus(X.transpose().times(X.times(theta).minus(y)).times(alpha / learningDataSet.size()));
		}
		
		parameters = theta.getColumnPackedCopy();
	}
	
	private Normalizer[] createNormalizers(List<LearningData> learningDataSet) {
		return IntStream.range(0, inputDimension).mapToObj(i -> new Normalizer(getColumnValues(learningDataSet, i))).toArray(Normalizer[]::new);
	}
	
	private double[] getColumnValues(List<LearningData> learningDataSet, int i) {
		return learningDataSet.stream().mapToDouble(learningData -> learningData.input.values[i]).toArray();
	}
	
	private Matrix convertToNormalizedMatrix(List<LearningData> learningDataSet, Normalizer[] normalizers) {
		double[] values = learningDataSet.stream().flatMapToDouble(learningData -> DoubleStream.concat(DoubleStream.of(1), DoubleStream.of(normalize(learningData.input.values, normalizers)))).toArray();
		return new Matrix(values, inputDimension+1).transpose();
	}
	
	private static double[] normalize(double[] values, Normalizer[] normalizers) {
		if(values.length != normalizers.length) throw new IllegalArgumentException();
		
		return IntStream.range(0, values.length).mapToDouble(i -> normalizers[i].normalize(values[i])).toArray();
	}
	
	private static Matrix convertToVector(List<LearningData> learningDataSet) {
		double[] values = learningDataSet.stream().mapToDouble(learningData -> learningData.output).toArray();
		return new Matrix(values, learningDataSet.size());
	}

	@Override
	public double calculateOutput(Input input) {
		checkDimension(input);
		return parameters[0] + IntStream.range(0, inputDimension).mapToDouble(i -> parameters[i+1] * normalizers[i].normalize(input.values[i])).sum();
	}

	@Override
	public double calculateCost(List<LearningData> learningDataSet) {
		int n = learningDataSet.size();
		return learningDataSet.stream().mapToDouble(learningData -> Math.square(learningData.output - calculateOutput(learningData.input))).sum() / (2 * n);
	}
	
	private void checkDimension(Input input) {
		if(input.dimension() != inputDimension) throw new IllegalArgumentException("Input dimension must be " + inputDimension);
	}
	
	private static class Normalizer {
		
		private final double average;
		private final double stdev;

		Normalizer(double[] values) {
			average = Statistics.average(values);
			stdev = Statistics.stDev(values);
		}
		
		double normalize(double value) {
			return (value - average) / stdev;
		}
	}

}
