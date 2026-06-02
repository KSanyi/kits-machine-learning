package kits.ml.neuralnet;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import kits.ml.core.LearningData;
import kits.ml.core.math.MLMath;
import kits.ml.core.math.linalg.Matrix;
import kits.ml.core.math.linalg.Vector;

public class VectorizedNeuralNet {

    private final Matrix[] weightMatrixes;

    private final Vector[] biases;

    private final double learningRate;
    
    private final Random random;

    public VectorizedNeuralNet(int ... neuronsInLayers) {
        this(0.1, new Random(), neuronsInLayers);
    }

    public VectorizedNeuralNet(double learningRate, Random random, int ... neuronsInLayers) {
        this.learningRate = learningRate;
        this.random = random;
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
    
    public void learn(List<LearningData> trainingBatch) {
        int n = weightMatrixes.length;
        
        List<Gradient> gradientList = trainingBatch.parallelStream().map(trainingData -> calculateGradient(n, trainingData)).toList();
        
        Gradient gradient = calculateAverage(gradientList);
        
        // Gradient descent: update weights and biases
        for (int i = 0; i < n; i++) {
            // dW[l] = delta[l] ⊗ activations[l]  (outer product)
            Matrix dW = gradient.weightMatrixGradients[i];
            Vector db = gradient.biasGradients[i];
            dW.scaleThis(learningRate);
            db.scaleThis(learningRate);
            weightMatrixes[i].minusThis(dW);
            biases[i].minusThis(db);
        }
    }
    
    private Gradient calculateGradient(int n, LearningData trainingData) {
        
        Vector input = trainingData.input();
        Vector trueOutput = Vector.createOneHot(10, (int)trainingData.output());
        
        Vector[] activations = new Vector[n + 1];
        Vector[] preActivations = new Vector[n];

        activations[0] = input;
        for (int i = 0; i < n; i++) {
            preActivations[i] = weightMatrixes[i].multiply(activations[i]).plus(biases[i]);
            activations[i + 1] = MLMath.sigmoid(preActivations[i]);
        }

        // Backward pass: compute error deltas from output to input
        Vector[] dbs = new Vector[n];
        Matrix[] dWs = new Matrix[n];

        // using quadratic cost function
//        Vector error = activations[n].minus(trueOutput);
//        dbs[n - 1] = error.map(j -> error.get(j) * aOut.get(j) * (1 - aOut.get(j)));
//        dWs[n - 1] = dbs[n - 1].multiply(activations[n - 1]);
        
        // using cross entropy cost function:
         dbs[n - 1] = activations[n].minus(trueOutput);
         dWs[n - 1] = dbs[n - 1].multiply(activations[n - 1]);

        // Hidden layer deltas: delta[l] = (W[l+1]^T * delta[l+1]) ⊙ sigmoid'(z[l])
        // sigmoid'(z) = a*(1-a) where a = sigmoid(z) = activations[l+1]
        for (int i = n - 2; i >= 0; i--) {
            Vector wTDelta = weightMatrixes[i + 1].transpose().multiply(dbs[i + 1]);
            Vector sigmoidGrad = activations[i + 1].map(x -> x * (1 - x));
            dbs[i] = wTDelta.hadamardProduct(sigmoidGrad);
            dWs[i] = dbs[i].multiply(activations[i]);
        }
        
        return new Gradient(dWs, dbs);
    }
    
    private static Gradient calculateAverage(List<Gradient> gradientList) {
        Gradient avg = gradientList.get(0);
        int batchSize = gradientList.size();
        int n = avg.biasGradients.length;
        
        for(int i=1;i<gradientList.size();i++) {
            Gradient gradient = gradientList.get(i);
            for(int j=0;j<n;j++) {
                avg.weightMatrixGradients[j].plusThis(gradient.weightMatrixGradients[j]);
                avg.biasGradients[j].plusThis(gradient.biasGradients[j]);
            }
        }
        
        avg.scaleThis(1.0 / batchSize);
        
        return avg;
    }
    
    private static record Gradient(Matrix[] weightMatrixGradients, Vector[] biasGradients) {
        void scaleThis(double factor) {
            for(int j=0;j<weightMatrixGradients.length;j++) {
                weightMatrixGradients[j].scaleThis(factor);
                biasGradients[j].scaleThis(factor);
            }
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
    
    public void save2(String filePath) throws IOException {
        try (PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(filePath)))) {
            out.println("learningRate=" + learningRate);
            out.println("layers=" + weightMatrixes.length);
            for (int l = 0; l < weightMatrixes.length; l++) {
                Matrix m = weightMatrixes[l];
                Vector b = biases[l];
                out.println();
                out.println("[layer " + (l + 1) + "]");
                out.println("rows=" + m.getNrRows());
                out.println("cols=" + m.getNrColumns());
                out.println("weights=");
                for (int r = 0; r < m.getNrRows(); r++) {
                    StringBuilder row = new StringBuilder();
                    for (int c = 0; c < m.getNrColumns(); c++) {
                        if (c > 0) row.append(' ');
                        row.append(String.format("%.8f", m.get(r, c)));
                    }
                    out.println(row);
                }
                out.println("biases=");
                out.println(b.stream()
                        .mapToObj(v -> String.format("%.8f", v))
                        .collect(Collectors.joining(" ")));
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

            VectorizedNeuralNet net = new VectorizedNeuralNet(learningRate, new Random(), neuronsInLayers);
            for (int l = 0; l < n; l++) {
                net.weightMatrixes[l] = loadedWeights[l];
                net.biases[l] = loadedBiases[l];
            }
            return net;
        }
    }

}
