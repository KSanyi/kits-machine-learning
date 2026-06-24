package kits.ml.core.math.optimization;

import java.util.function.Function;

import kits.ml.core.math.linalg.Vector;

public class GradientDescentWithMomentumOptimizer implements GradientOptimizer {

    private final int steps;
    private final double alpha;
    private final double beta;
    private final double tolerance;
    
    public GradientDescentWithMomentumOptimizer(int steps, double alpha, double beta, double tolerance) {
        this.steps = steps;
        this.alpha = alpha;
        this.tolerance = tolerance;
        this.beta = beta;
    }
    
    @Override
    public Vector optimize(Vector startingVector, Function<Vector, Double> function, Function<Vector, Vector> gradient) {
        
        Vector xVector = startingVector;
        
        Vector prevMomentum = new Vector(startingVector.length());
        double prevValue = Double.MAX_VALUE;
        for (int i = 0; i < steps; i++) {
            double currentValue = function.apply(xVector);
            if(prevValue - currentValue < tolerance) {
                System.out.println("Stopped after " + i + " iteration");
                return xVector;
            }
            System.out.println(xVector + ": " + currentValue);
            
            Vector gradientVector = gradient.apply(xVector);
            Vector momentum = gradientVector.plus(prevMomentum.scale(beta));
            prevMomentum = momentum;
            prevValue = currentValue;
            
            xVector = xVector.minus(momentum.scale(alpha));
        }
        
        return xVector;
    }

}
