package kits.ml.core.math.linalg;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;

import org.junit.jupiter.api.Test;

import kits.ml.core.math.linalg.Decomposition.LU;

public class GaussEliminationCalculatorTest {

    @Test
    public void test1() {
        
        Matrix A = new Matrix(
            new double[] {2, 4, 6},
            new double[] {1, 5, 9},
            new double[] {2, 1, 3}
        );
        
        Vector b = new Vector(4, 2, 7);
        
        Vector result = GaussEliminationCalculator.solveEquation(A, b);
        
        assertEquals(new Vector(3, -2, 1), result);
    }
    
    @Test
    public void test2() {
        
        Matrix A = new Matrix(
            new double[] { 2,  1, 1},
            new double[] { 4, -6, 0},
            new double[] {-2,  7, 2}
        );
        
        Vector b = new Vector(5, -2, 9);
        
        Vector result = GaussEliminationCalculator.solveEquation(A, b);
        
        assertEquals(new Vector(1, 1, 2), result);
    }
    
    @Test
    public void test3() {
        
        Matrix A = new Matrix(
            new double[] {1, 1, 1},
            new double[] {2, 2, 5},
            new double[] {4, 6, 8}
        );
        
        Vector b = new Vector(4, 2, 7);
        
        Vector result = GaussEliminationCalculator.solveEquation(A, b);
        
        assertEquals(new Vector(6.5, -0.5, -2), result);
    }
    
    @Test
    public void testRandomEquationsWithIntegerCoefficients() {
        
        for(int i=0;i<100;i++) {
            Matrix A = MatrixUtils.generateRandomIntMatrix(100, 100, 100);
            Vector x = MatrixUtils.generateRandomIntVector(100, 100);
            Vector b = A.multiply(x);
            Vector result = GaussEliminationCalculator.solveEquation(A, b);
            
            assertEquals(x, result);
        }
    }
    
    @Test
    public void testRandomEquationsWithDoubleCoefficients() {
        
        Random random = new Random();
        
        for(int i=0;i<100;i++) {
            Matrix A = MatrixUtils.generateRandomMatrix(100, 100, () -> random.nextDouble() * 100);
            Vector x = MatrixUtils.generateRandomVector(100, () -> random.nextDouble() * 100);
            Vector b = A.multiply(x);
            Vector result = GaussEliminationCalculator.solveEquation(A, b);
            
            assertEquals(x, result);
        }
    }
    
    @Test
    public void testInverse() {
        
        Matrix A = new Matrix(
            new double[] { 2,  1, 1},
            new double[] { 4, -6, 0},
            new double[] {-2,  7, 2}
        );
        
        Matrix inverse = GaussEliminationCalculator.calculateInverse(A);
        
        assertEquals(Matrix.createIdentity(3), A.multiply(inverse));
    }
    
    @Test
    public void testLU1() {
        
        Matrix A = new Matrix(
            new double[] { 1,  4, -3},
            new double[] {-2,  8,  5},
            new double[] { 3,  4,  7}
        );
        
        LU luDecomposition = GaussEliminationCalculator.createLUDecomposition(A);

        Matrix P = luDecomposition.P();
        Matrix L = luDecomposition.L();
        Matrix U = luDecomposition.U();
        
        assertEquals(P.multiply(A), L.multiply(U));
        
        Matrix expectedP = new Matrix(
                new double[] {0, 0, 1},
                new double[] {0, 1, 0},
                new double[] {1, 0, 0}
            );
        
        Matrix expectedL = new Matrix(
                new double[] { 1,     0,    0},
                new double[] {-0.667, 1,    0},
                new double[] { 0.333, 0.25, 1}
            );
        
        Matrix expectedU = new Matrix(
                new double[] {3,  4,     7   },
                new double[] {0, 10.667, 9.667},
                new double[] {0,  0,    -7.75}
            );
        
        assertEquals(expectedP, P);
        assertEquals(expectedL, L);
        assertEquals(expectedU, U);
    }
    
    @Test
    public void testLU2() {
        
        Matrix A = new Matrix(
            new double[] {1, 1, 1},
            new double[] {0, 0, 1},
            new double[] {2, 3, 4}
        );
        
        LU luDecomposition = GaussEliminationCalculator.createLUDecomposition(A);
        
        Matrix P = luDecomposition.P();
        Matrix L = luDecomposition.L();
        Matrix U = luDecomposition.U();
        
        assertEquals(P.multiply(A), L.multiply(U));
        
        Matrix expectedP = new Matrix(
                new double[] {0, 0, 1},
                new double[] {1, 0, 0},
                new double[] {0, 1, 0}
            );
        
        Matrix expectedL = new Matrix(
                new double[] {1,   0, 0},
                new double[] {0.5, 1, 0},
                new double[] {0,   0, 1}
            );
        
        Matrix expectedU = new Matrix(
                new double[] {2,  3,    4},
                new double[] {0, -0.5, -1},
                new double[] {0,  0,    1}
            );
        
        assertEquals(expectedP, P);
        assertEquals(expectedL, L);
        assertEquals(expectedU, U);
    }
    
    @Test
    public void testLUP2() {
        
        Matrix A = new Matrix(
            new double[] {1, 2, 1},
            new double[] {1, 2, 2},
            new double[] {2, 1, 1}
        );
        
        LU luDecomposition = GaussEliminationCalculator.createLUDecomposition(A);
        
        Matrix L = luDecomposition.L();
        Matrix U = luDecomposition.U();
        Matrix P = luDecomposition.P();
        
        assertEquals(P.multiply(A), L.multiply(U));
        
        Matrix expectedP = new Matrix(
                new double[] {0, 0, 1},
                new double[] {0, 1, 0},
                new double[] {1, 0, 0}
            );
        
        Matrix expectedL = new Matrix(
                new double[] {1,   0, 0},
                new double[] {0.5, 1, 0},
                new double[] {0.5, 1, 1}
            );
        
        Matrix expectedU = new Matrix(
                new double[] {2, 1,   1},
                new double[] {0, 1.5, 1.5},
                new double[] {0, 0,  -1}
            );
        
        assertEquals(expectedP, P);
        assertEquals(expectedL, L);
        assertEquals(expectedU, U);
        
    }
    
    @Test
    public void testLUP3() {
            
        Matrix A = new Matrix(
            new double[] {2, 1, 1, 0},
            new double[] {4, 3, 3, 1},
            new double[] {8, 7, 9, 5},
            new double[] {6, 7, 9, 8}
        );
        
        LU luDecomposition = GaussEliminationCalculator.createLUDecomposition(A);
        
        Matrix L = luDecomposition.L();
        Matrix U = luDecomposition.U();
        Matrix P = luDecomposition.P();
        
        assertEquals(P.multiply(A), L.multiply(U));
    }
    
}
