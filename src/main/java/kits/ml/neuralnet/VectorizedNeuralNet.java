package kits.ml.neuralnet;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Random;

import kits.ml.core.math.MLMath;
import kits.ml.core.math.linalg.Matrix;
import kits.ml.core.math.linalg.Vector;

public class VectorizedNeuralNet {

    private final Matrix[] weightMatrixes;

    private final Vector[] biases;

    private final double learningRate;

    public VectorizedNeuralNet(int ... neuronsInLayers) {
        this(0.1, neuronsInLayers);
    }

    public VectorizedNeuralNet(double learningRate, int ... neuronsInLayers) {
        this.learningRate = learningRate;
        weightMatrixes = new Matrix[neuronsInLayers.length-1];
        biases = new Vector[neuronsInLayers.length-1];
        for(int i=1;i<neuronsInLayers.length;i++) {
            int neuronsInLayer = neuronsInLayers[i];
            int neuronsInPrevLayer = neuronsInLayers[i-1];
            weightMatrixes[i-1] = new Matrix(neuronsInLayer, neuronsInPrevLayer);
            biases[i-1] = Vector.createZero(neuronsInLayer);
        }
    }

    public void randomizeWeights() {
        Random random = new Random(0);
        
        for(Matrix matrix : weightMatrixes) {
            // Xavier initialization
            double sigma = MLMath.sqrt(1.0 / (matrix.getNrColumns() + matrix.getNrRows()));
            matrix.apply((i, j) -> random.nextGaussian() * sigma);
        }
        
        for(Vector biasVector : biases) {
            biasVector.apply(i -> random.nextDouble(-0.01, 0.01));
        }
    }

    public Vector predict(Vector input) {
        Vector vector = input;
        for(int i=0;i<weightMatrixes.length;i++) {
            Matrix matrix = weightMatrixes[i];
            vector = matrix.multiply(vector).plus(biases[i]);
            vector = MLMath.sigmoid(vector);
        }
        //System.out.println("Predicted: " + vector);
        return vector;
    }
    
    public void learn(Vector input, Vector trueOutputVector) {
        int n = weightMatrixes.length;
        
        // Forward pass: store pre-activations z[l] and activations a[l] for each layer
        Vector[] activations = new Vector[n + 1];
        Vector[] preActivations = new Vector[n];

        activations[0] = input;
        for (int i = 0; i < n; i++) {
            preActivations[i] = weightMatrixes[i].multiply(activations[i]).plus(biases[i]);
            activations[i + 1] = MLMath.sigmoid(preActivations[i]);
        }

        // Backward pass: compute error deltas from output to input
        Vector[] deltas = new Vector[n];

        // Output layer delta for binary cross-entropy + sigmoid: delta = a_out - y
        deltas[n - 1] = activations[n].minus(trueOutputVector);

        // Hidden layer deltas: delta[l] = (W[l+1]^T * delta[l+1]) ⊙ sigmoid'(z[l])
        for (int i = n - 2; i >= 0; i--) {
            Vector wTDelta = weightMatrixes[i + 1].transpose().multiply(deltas[i + 1]);
            Vector sigmoidGrad = MLMath.sigmoidGradient(activations[i + 1]);
            deltas[i] = wTDelta.map(j -> wTDelta.get(j) * sigmoidGrad.get(j));
        }

        // Gradient descent: update weights and biases
        for (int i = 0; i < n; i++) {
            // dW[l] = delta[l] ⊗ activations[l]  (outer product)
            deltas[i].scaleThis(learningRate);
            Matrix dW = deltas[i].multiply(activations[i]);
            weightMatrixes[i].minusThis(dW);
            biases[i].minusThis(deltas[i]);
        }
    }

    public void save(String filePath) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filePath)))) {
            out.writeDouble(learningRate);
            out.writeInt(weightMatrixes.length);
            for (int l = 0; l < weightMatrixes.length; l++) {
                Matrix m = weightMatrixes[l];
                out.writeInt(m.getNrRows());
                out.writeInt(m.getNrColumns());
                for (int r = 0; r < m.getNrRows(); r++) {
                    for (int c = 0; c < m.getNrColumns(); c++) {
                        out.writeDouble(m.get(r, c));
                    }
                }
                Vector b = biases[l];
                for (int i = 0; i < b.length(); i++) {
                    out.writeDouble(b.get(i));
                }
            }
        }
    }

    public static VectorizedNeuralNet load(String filePath) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)))) {
            double learningRate = in.readDouble();
            int n = in.readInt();

            Matrix[] loadedWeights = new Matrix[n];
            Vector[] loadedBiases = new Vector[n];

            for (int l = 0; l < n; l++) {
                int rows = in.readInt();
                int cols = in.readInt();
                double[][] values = new double[rows][cols];
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        values[r][c] = in.readDouble();
                    }
                }
                loadedWeights[l] = new Matrix(values);

                double[] biasValues = new double[rows];
                for (int i = 0; i < rows; i++) {
                    biasValues[i] = in.readDouble();
                }
                loadedBiases[l] = new Vector(biasValues);
            }

            // Reconstruct the layer sizes from the matrix dimensions
            int[] neuronsInLayers = new int[n + 1];
            neuronsInLayers[0] = loadedWeights[0].getNrColumns();
            for (int l = 0; l < n; l++) {
                neuronsInLayers[l + 1] = loadedWeights[l].getNrRows();
            }

            VectorizedNeuralNet net = new VectorizedNeuralNet(learningRate, neuronsInLayers);
            for (int l = 0; l < n; l++) {
                net.weightMatrixes[l] = loadedWeights[l];
                net.biases[l] = loadedBiases[l];
            }
            return net;
        }
    }

}
