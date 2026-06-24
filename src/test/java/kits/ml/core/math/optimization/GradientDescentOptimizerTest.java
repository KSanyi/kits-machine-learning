package kits.ml.core.math.optimization;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.function.Function;

import org.junit.jupiter.api.Test;

import kits.ml.core.math.MLMath;
import kits.ml.core.math.linalg.GaussEliminationCalculator;
import kits.ml.core.math.linalg.Matrix;
import kits.ml.core.math.linalg.Vector;

class GradientDescentOptimizerTest {

    private static final double TOLERANCE = 0.001;
    
    //@Test
    void test() {
       GradientDescentOptimizer optimizer = new GradientDescentOptimizer(100, 0.1, 0.001);
       
       Matrix A = new Matrix(new double[][] {
           {3, 1, 0},
           {1, 2, 1},
           {0, 1, 2}});
       
       Vector b = new Vector(1, 2, 3);
       
       // v' * A * v + v' * b
       Function<Vector, Double> function = v -> A.multiply(v).scalarProduct(v) + v.scalarProduct(b);
       Function<Vector, Vector> gradient = v -> A.multiply(v).scale(2).plus(b);
       
       Vector solutionFound = optimizer.optimize(new Vector(1, 2, 3), function, gradient);
       
       Vector explicitSolution = GaussEliminationCalculator.solveEquation(A.scale(2), b.scale(-1));
       
       assertEquals(0, solutionFound.minus(explicitSolution).norm(), TOLERANCE);
    }
    
    @Test
    void test2() {
        int steps = 10;
        double alpha = 0.1;
        GradientDescentOptimizer optimizer = new GradientDescentOptimizer(steps, alpha, 0.001);
       
        double b = 10;
       
        Function<Vector, Double> function = v -> 0.5 * MLMath.square(v.get(0)) + b * MLMath.square(v.get(1));  
        Function<Vector, Vector> gradient = v -> new Vector(v.get(0), b * v.get(1));
       
        Vector solutionFound = optimizer.optimize(new Vector(b, 1), function, gradient);
       
        Vector explicitSolution = new Vector(2);
       
        assertEquals(0, solutionFound.minus(explicitSolution).norm(), TOLERANCE);
    }
    
    @Test
    void test3() {
        int steps = 10;
        double alpha = 0.2317;
        double beta = 0.2702;
        GradientDescentWithMomentumOptimizer optimizer = new GradientDescentWithMomentumOptimizer(steps, alpha, beta, 0.0001);
           
        double b = 10;
           
        Function<Vector, Double> function = v -> 0.5 * MLMath.square(v.get(0)) + b * MLMath.square(v.get(1));  
        Function<Vector, Vector> gradient = v -> new Vector(v.get(0), b * v.get(1));
           
        Vector solutionFound = optimizer.optimize(new Vector(b, 1), function, gradient);
           
        Vector explicitSolution = new Vector(2);
           
        assertEquals(0, solutionFound.minus(explicitSolution).norm(), TOLERANCE);
    }

}
