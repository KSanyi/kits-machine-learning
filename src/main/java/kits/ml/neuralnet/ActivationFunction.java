package kits.ml.neuralnet;

import kits.ml.core.math.MLMath;

public interface ActivationFunction {
    
    public double apply(double value);

    // derivative of the activation with respect to its pre-activation input, expressed in terms of the already-activated value
    public double derivative(double activatedValue);
    
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
            
            public double derivative(double activatedValue) {
                return activatedValue > 0 ? 1 : 0;
            }
        };

    }

}

