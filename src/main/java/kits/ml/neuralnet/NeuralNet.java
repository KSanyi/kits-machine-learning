package kits.ml.neuralnet;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import kits.ml.core.Input;

public class NeuralNet {

	private final List<Layer> hiddenLayers;
	
	public NeuralNet(List<double[][]> layerWeights) {
		hiddenLayers = new ArrayList<>();
		for(double[][] weights : layerWeights) {
			hiddenLayers.add(new Layer(weights));
		}
	}
	
	public NeuralNet(int inputDimension, int ... hiddenLayerNeurons) {
		if(hiddenLayerNeurons.length == 0) throw new IllegalArgumentException("At least 1 layer is expected");
		
		hiddenLayers = new ArrayList<>();
		int numberOfNeuronsInPrevLayer = inputDimension;
		for(int numberOfNeurons : hiddenLayerNeurons) {
			hiddenLayers.add(new Layer(numberOfNeurons, numberOfNeuronsInPrevLayer));
			numberOfNeuronsInPrevLayer = numberOfNeurons;
		}
	}
	
	public int predict(Input input) {
		Input inp = input;
		double[] output = null;
		for(Layer hiddenLayer : hiddenLayers) {
			output = hiddenLayer.calculateOutput(inp);
			inp = new Input(output);
		}
		
		return findIndexForMaxOutput(output) + 1;
	}
	
	private int findIndexForMaxOutput(double[] output) {
		double max = 0;
		int indexForMax = -1;
		for(int i=0;i<output.length;i++) {
			//System.out.println("Output for " + (i+1) + " " + output[i]);
		    if(output[i] > max) {
				max = output[i];
				indexForMax = i;
			}
		}
		return indexForMax;
	}
	
	private static class Layer {
		
		final List<Neuron> neurons;
		
		Layer(int numberOfNeurons, int numberOfNeuronsInPrevLayer) {
			neurons = IntStream.range(0, numberOfNeurons).mapToObj(i -> new Neuron(numberOfNeuronsInPrevLayer)).collect(Collectors.toList());
		}
		
		Layer(double[][] weights) {
			neurons = IntStream.range(0, weights.length).mapToObj(i -> new Neuron(weights[i])).collect(Collectors.toList());
		}
		
		public double[] calculateOutput(Input input) {
			return neurons.stream().mapToDouble(neuron -> neuron.calculateOutput(input)).toArray();
		}
	}
	
}
