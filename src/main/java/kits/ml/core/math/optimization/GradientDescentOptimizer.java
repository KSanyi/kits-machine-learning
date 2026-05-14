package kits.ml.core.math.optimization;

import java.util.function.Function;

import kits.ml.core.math.linalg.Vector;

public class GradientDescentOptimizer implements GradientOptimizer {

    private final int steps;
    private final double alpha;
    private final double tolerance;
    
    public GradientDescentOptimizer(int steps, double alpha, double tolerance) {
        this.steps = steps;
        this.alpha = alpha;
        this.tolerance = tolerance;
    }
    
    @Override
    public Vector optimize(Vector startingVector, Function<Vector, Double> function, Function<Vector, Vector> gradient) {
        
        Vector xVector = startingVector;
        
        for (int i = 0; i < steps; i++) {
            double currentValue = function.apply(xVector);
            System.out.println(xVector + ": " + currentValue);
            
            Vector gradientVector = gradient.apply(xVector);
            if(gradientVector.norm() < tolerance) {
                System.out.println("Stopped after " + i + " iteration");
                return xVector;
            }
            
            xVector = xVector.minus(gradientVector.scale(alpha));
        }
        
        return xVector;
    }

}
