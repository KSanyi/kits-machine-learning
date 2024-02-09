package kits.ml.core.math.linalg;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import kits.ml.core.math.linalg.Decomposition.QR;

@SuppressWarnings("static-method")
public class GramSchmidtCalculatorTest {

    private static final double EPSILON = 0.001;
    
    @Test
    public void test() {
        
        Matrix A = new Matrix(new double[][] {
            {1, 2,  1},
            {1, 1, -1},
            {2, 1,  2}}
        );
        
        Matrix Q = new Matrix(new double[][] {
            {0.408,  0.862,  0.302},
            {0.408,  0.123, -0.905},
            {0.816, -0.492,  0.302}}
        );
        
        assertEquals(Q, GramSchmidtCalculator.createOrtoNormalBaseMatrix(A));
        
        assertOrtonormal(Q);
    }
    
    private static void assertOrtonormal(Matrix A) {
        assertEquals(0, A.getColumnVector(0).scalarProduct(A.getColumnVector(1)), EPSILON);
        assertEquals(0, A.getColumnVector(0).scalarProduct(A.getColumnVector(2)), EPSILON);
        assertEquals(0, A.getColumnVector(1).scalarProduct(A.getColumnVector(2)), EPSILON);
        
        assertEquals(1, A.getColumnVector(0).norm(), EPSILON);
        assertEquals(1, A.getColumnVector(1).norm(), EPSILON);
        assertEquals(1, A.getColumnVector(2).norm(), EPSILON);
    }
    
    @Test
    public void testQR() {
        
        Matrix A = new Matrix(new double[][] {
            {1, 2,  1},
            {1, 1, -1},
            {2, 1,  2}}
        );
        
        QR qr = GramSchmidtCalculator.createQRDecomposition(A);
        
        
        Matrix Q = new Matrix(new double[][] {
            {0.408,  0.862,  0.302},
            {0.408,  0.123, -0.905},
            {0.816, -0.492,  0.302}}
        );
        
        Matrix R = new Matrix(new double[][] {
            {2.449, 2.041,  1.633},
            {0,     1.354, -0.246},
            {0,     0,      1.809}}
        );
        
        assertEquals(Q, qr.Q());
        assertEquals(R, qr.R());
        
        assertEquals(A, qr.Q().multiply(qr.R()));
        
        assertOrtonormal(Q);
    }
    
    @Test
    public void testDegenerate() {
        
        Matrix A = new Matrix(new double[][] {
            {1, 1, 0},
            {1, 1, 1},
            {2, 2, 2}}
        );
        
        Assertions.assertThrows(IllegalArgumentException.class, () -> {
            GramSchmidtCalculator.createOrtoNormalBaseMatrix(A);
        });
        
    }
    
}
