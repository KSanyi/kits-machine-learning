# kits-machine-learning

A Java neural network and ML library built from scratch for educational purposes. The flagship application is a **handwritten digit recognizer** trained on MNIST-format PNG images.

---

## How It Works

### Neural Network

The core is a fully-connected feedforward network with mini-batch stochastic gradient descent and backpropagation.

```java
// hidden layers use ReLU; output layer uses Sigmoid (required by cross-entropy)
NeuralNet net = new NeuralNet(
    StandardCostFunction.CROSS_ENTROPY,
    StandardActivationFunction.RELU,    // hidden activation
    StandardActivationFunction.SIGMOID, // output activation
    784,   // input: 28×28 pixels flattened
    256,   // hidden layer 1
    128,   // hidden layer 2
    10     // output: one neuron per digit (0–9)
);
```

**Training loop (per epoch):**
1. Shuffle training data
2. For each sample: forward pass → compute gradient → update weights & biases
3. Log cost

**Weight initialization:** He — `σ = √(2 / nrInputs)`, biases in `[-0.01, 0.01]`. He initialization is optimal for ReLU; Xavier underestimates variance and leads to dead neurons early in training.

### Digit Detection Pipeline

1. Load PNG images from `train/<digit>/*.png`
2. Convert to greyscale: `(R+G+B) / 3`
3. Normalize: `greyscale / 255.0 - 0.5` (centers around −0.5)
4. Flatten to a 784-element vector
5. Train for N epochs, then run inference on test images
6. Classify via `argmax` of the 10 output neurons

### Activation Functions

| Name | Formula | Derivative (from activated value `a`) | Typical use |
|------|---------|---------------------------------------|-------------|
| `RELU` | `max(0, x)` | `1 if a > 0, else 0` | Hidden layers |
| `SIGMOID` | `1 / (1 + e^-x)` | `a * (1 - a)` | Output layer (required by cross-entropy) |
| `NONE` | `x` | `1` | Passthrough / regression output |

> **ReLU + Sigmoid split:** the `NeuralNet` constructor accepts separate hidden and output activation functions. Always pair `RELU` hidden layers with a `SIGMOID` output when using `CROSS_ENTROPY` — cross-entropy requires outputs in (0, 1), which ReLU cannot guarantee. Using `RELU` everywhere causes `log(negative)` → NaN in the cost.

### Cost Functions

| Name | Use case | Gradient |
|------|----------|----------|
| `CROSS_ENTROPY` | Classification | `(a - y) / (a * (1 - a))` |
| `QUADRATIC` | Regression | `2 * (a - y)` |

> **Use `CROSS_ENTROPY` for digit classification.** It avoids the "learning slowdown" of quadratic loss when the network output is saturated.

---

## Getting Above 90% Accuracy

### 1. Activation Function — ReLU hidden, Sigmoid output

ReLU (`max(0, x)`) avoids the vanishing gradient problem that slows Sigmoid in deep networks. The output layer must stay Sigmoid so cross-entropy receives values in (0, 1).

```java
new NeuralNet(CROSS_ENTROPY, RELU, SIGMOID, 784, 256, 128, 10);
```

### 2. Weight Initialization — He instead of Xavier

Xavier (`σ = √(1 / n)`) underestimates variance for ReLU and leads to dead neurons early in training. He initialization (`σ = √(2 / nrInputs)`) is the correct choice and is now the default.

### 3. Network Architecture

| Configuration | Samples/digit | Epochs | Est. runtime (local CPU) | Expected accuracy |
|--------------|--------------|--------|--------------------------|------------------|
| 784 → 60 → 10, Sigmoid | 100 | 100 | ~1 min | ~85–88% |
| 784 → 128 → 10, ReLU+Sigmoid | 100 | 100 | ~2 min | ~90% |
| **784 → 256 → 128 → 10, ReLU+Sigmoid** | **100** | **100** | **~3 min** | **~92–94%** |
| 784 → 256 → 128 → 10, ReLU+Sigmoid | all (~6 000) | 300 | ~2–5 hours | ~97%+ |

> **Local runtime budget:** this project targets a **3-minute** training cap on a standard laptop (pure Java, no GPU). The current configuration (100 samples/digit, 100 epochs, 784→256→128→10) sits at that limit. Increasing samples or epochs beyond this requires patience — the bottleneck is pure Java matrix multiplication with no hardware acceleration.

- A second hidden layer (256→128) lets the network learn more abstract features without blowing the time budget
- More training data is the single biggest lever for pushing past 95%, but only feasible if runtime is not a constraint

### 4. Training Hyperparameters

| Parameter | Baseline | Current (3-min budget) | Unconstrained |
|-----------|----------|----------------------|---------------|
| Architecture | 784 → 128 → 10 | 784 → 256 → 128 → 10 | 784 → 256 → 128 → 10 |
| Samples per digit | 100 | 100 | all (~6 000) |
| Epochs | 100 | 100 | 300 |
| Learning rate | 0.01 | 0.01 | 0.01 |

Watch the cost log — if it plateaus early, reduce the learning rate to 0.001.

### 5. Data Normalization

`pixel / 255.0 - 0.5` centers inputs around −0.5. Keep this consistent between train and test — do not change the scaling after training.

### 6. Cost Function

Always use `CROSS_ENTROPY` for classification. It provides cleaner gradients than `QUADRATIC` when output activations are near 0 or 1, and pairs naturally with the Sigmoid output layer.

---

## Project Structure

```
src/main/java/kits/ml/
├── application/
│   ├── DigitDetector.java       # loads images, trains, predicts
│   └── DigitDetectorMain.java   # entry point
├── neuralnet/
│   ├── NeuralNet.java           # forward pass, backprop, SGD
│   ├── ActivationFunction.java  # interface + StandardActivationFunction enum
│   └── CostFunction.java        # interface + StandardCostFunction enum
├── regression/
│   ├── LinearRegressionModel.java
│   ├── SimpleLinearRegressionModel.java
│   ├── LogisticRegression.java
│   └── MultiClassificationModel.java
└── core/math/
    ├── linalg/
    │   ├── Vector.java
    │   ├── Matrix.java
    │   ├── GaussEliminationCalculator.java
    │   └── CholeskyDecompositionCalculator.java
    └── optimization/
        ├── GradientDescentOptimizer.java
        └── GradientDescentWithMomentumOptimizer.java
```

---

## Quick Start

1. Download the [MNIST PNG dataset](https://github.com/myleott/mnist_png) and place it under a directory, e.g. `mnist-png/`.
2. Set the path in `DigitDetectorMain` to point to your dataset root.
3. Build and run:

```bash
mvn package
java -cp target/machine-learning-*.jar kits.ml.application.DigitDetectorMain
```

The application logs the cost after each epoch and prints the predicted digit for test images.

---

## Dependencies

- Java 25, Maven 3
- Apache Commons Lang 3.10
- SLF4J 1.7.30
- JUnit 5 (tests)
