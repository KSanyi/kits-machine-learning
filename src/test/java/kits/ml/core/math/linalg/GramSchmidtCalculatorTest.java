package kits.ml.core.math.linalg;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class GramSchmidtCalculatorTest {

    private static final double EPSILON = 0.001;
    
    @Test
    public void test() {
        
        Matrix A = new Matrix(
            new double[] {1, 2,  1},
            new double[] {1, 1, -1},
            new double[] {2, 1,  2}
        );
        
        Matrix Q = new Matrix(
            new double[] {0.408,  0.862,  0.302},
            new double[] {0.408,  0.123, -0.905},
            new double[] {0.816, -0.492,  0.302});
        
        assertEquals(Q, GramSchmidtCalculator.createOrtoNormalBaseMatrix(A));
        
        assertEquals(0, Q.getColumnVector(0).scalarProduct(Q.getColumnVector(1)), EPSILON);
        assertEquals(0, Q.getColumnVector(0).scalarProduct(Q.getColumnVector(2)), EPSILON);
        assertEquals(0, Q.getColumnVector(1).scalarProduct(Q.getColumnVector(2)), EPSILON);
        
        assertEquals(1, Q.getColumnVector(0).norm(), EPSILON);
        assertEquals(1, Q.getColumnVector(1).norm(), EPSILON);
        assertEquals(1, Q.getColumnVector(2).norm(), EPSILON);
    }
    
    @Test
    public void testDegenerate() {
        
        Matrix A = new Matrix(
            new double[] {1, 1, 0},
            new double[] {1, 1, 1},
            new double[] {2, 2, 2}
        );
        
        Assertions.assertThrows(IllegalArgumentException.class, () -> {
            GramSchmidtCalculator.createOrtoNormalBaseMatrix(A);
        });
        
    }
    
}
