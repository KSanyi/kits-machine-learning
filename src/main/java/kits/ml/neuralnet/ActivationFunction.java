package kits.ml.neuralnet;

import kits.ml.core.math.MLMath;
import kits.ml.core.math.linalg.Vector;

public interface ActivationFunction {

    public double apply(double value);

    // derivative of the activation with respect to its pre-activation input, expressed in terms of the already-activated value
    public double derivative(double activatedValue);

    // apply activation to a full vector; override for non-element-wise activations (e.g. softmax)
    default Vector applyVector(Vector v) {
        return v.map(this::apply);
    }

    // combined output-layer error: costGradient ⊙ activationDerivative
    // override when the activation+cost pair has a simpler closed-form (e.g. softmax + cross-entropy -> a - y)
    default Vector outputDelta(Vector activatedOutput, Vector target, CostFunction costFunction) {
        Vector costGradient = costFunction.gradient(activatedOutput, target);
        Vector activationDerivative = activatedOutput.map(this::derivative);
        return costGradient.hadamardProduct(activationDerivative);
    }

    public enum StandardActivationFunction implements ActivationFunction {

        NONE {
            public double apply(double value) {
                return value;
            }

            public double derivative(double activatedValue) {
                return 1;
            }
        },

        SIGMOID {
            public double apply(double value) {
                return MLMath.sigmoid(value);
            }

            public double derivative(double activatedValue) {
                return activatedValue * (1 - activatedValue);
            }
        },

        RELU {
            public double apply(double value) {
                return Math.max(0, value);
            }

            // derivative expressed via activated value: relu'(x) = 1 iff relu(x) > 0
            public double derivative(double activatedValue) {
                return activatedValue > 0 ? 1 : 0;
            }
        },

        SOFTMAX {
            public double apply(double value) {
                throw new UnsupportedOperationException("Softmax is not element-wise; use applyVector()");
            }

            public double derivative(double activatedValue) {
                throw new UnsupportedOperationException("Softmax Jacobian is a matrix; use outputDelta() instead");
            }

            // softmax: e^zᵢ / Σe^zⱼ  (subtract max for numerical stability)
            @Override
            public Vector applyVector(Vector v) {
                double max = v.stream().max().orElse(0);
                double[] exps = v.stream().map(x -> Math.exp(x - max)).toArray();
                double sum = 0;
                for (double e : exps) sum += e;
                double[] result = new double[exps.length];
                for (int i = 0; i < exps.length; i++) result[i] = exps[i] / sum;
                return new Vector(result);
            }

            // softmax + cross-entropy gradient simplifies to (a - y), avoiding the full Jacobian
            @Override
            public Vector outputDelta(Vector activatedOutput, Vector target, CostFunction costFunction) {
                return activatedOutput.minus(target);
            }
        };

    }

}

