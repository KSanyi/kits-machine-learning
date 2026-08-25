package kits.ml.neuralnet;

import kits.ml.core.math.linalg.Vector;

public interface CostFunction {
    
    double cost(Vector predictedOutput, Vector trueOutput);

    // gradient of the cost with respect to the predicted (activated) output
    Vector gradient(Vector predictedOutput, Vector trueOutput);
    
    public enum StandardCostFunction implements CostFunction {

        QUADRATIC {
            public double cost(Vector predictedOutput, Vector trueOutput) {
                return predictedOutput.minus(trueOutput).normSquared();
            }
            
            public Vector gradient(Vector predictedOutput, Vector trueOutput) {
                return predictedOutput.minus(trueOutput).scale(2);
            }
        },

        CROSS_ENTROPY {
            public double cost(Vector predictedOutput, Vector trueOutput) {
                double sum = 0;
                for(int i=0;i<trueOutput.length();i++) {
                    double y = trueOutput.get(i);
                    double activatedValue = predictedOutput.get(i);
                    sum += y * Math.log(activatedValue);
                }
                return -sum;
            }
            
            public Vector gradient(Vector predictedOutput, Vector trueOutput) {
                double[] result = new double[predictedOutput.length()];
                for(int i=0;i<result.length;i++) {
                    double y = trueOutput.get(i);
                    double activatedValue = predictedOutput.get(i);
                    
                    result[i] = -y / activatedValue;
                }
                return new Vector(result);
            }
        };

    }
}


