package kits.ml.core.math.linalg;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

public class MatrixTest {

    @Test
    public void basic() {
        
        Matrix A = new Matrix(2, 2);
        A.set(1, 1, 3.0);
        
        Matrix expected = new Matrix(new double[][] {
                                        {0, 0},
                                        {0, 3}});
        
        assertEquals(expected, A);
    }
    
    @Test
    public void fromColumnVectors() {
        
        Matrix A = MatrixFactory.fromColumnVectors(new Vector(1, 2, 3), new Vector(4, 5, 6));
        
        Matrix expected = new Matrix(new double[][] {
                                        {1, 4},
                                        {2, 5},
                                        {3, 6}});
        
        assertEquals(expected, A);
    }
    
    @Test
    public void equality() {
        
        Matrix A = new Matrix(new double[][] {
                                    {1, 2},
                                    {3, 4}});
        
        Matrix B = new Matrix(new double[][] {
                                    {1, 2},
                                    {3, 4}});
        
        assertTrue(A.equals(B));
        
        A.set(0, 0, 1.001);
        assertTrue(A.equals(B));
        
        A.set(0, 0, 1.01);
        assertFalse(A.equals(B));
    }
    
    @Test
    public void transpose() {
        
        Matrix A = new Matrix(new double[][] {
            {1, 2, 3},
            {4, 5, 6}});
        
        Matrix expectedTranspose = new Matrix(new double[][] {
                {1, 4},
                {2, 5},
                {3, 6}});
        
        assertEquals(expectedTranspose, A.transpose());
    }
    
    @Test
    public void map() {
        
        Matrix A = new Matrix(new double[][] {
            {1, 2},
            {3, 4}});
        
        Matrix expected = new Matrix(new double[][] {
                {2, 4},
                {6, 8}});
        
        assertEquals(expected, A.map((i,j) -> A.get(i, j) * 2.0));
    }
    
    @Test
    public void copy() {
        
        Matrix A = new Matrix(new double[][] {
                {1, 2},
                {3, 4}});
        
        Matrix B = new Matrix(A);
        
        B.set(0, 0, -1);
        B.set(0, 1, -2);
        B.set(1, 0, -3);
        B.set(1, 1, -4);
        
        assertEquals(1, A.get(0, 0));
        assertEquals(2, A.get(0, 1));
        assertEquals(3, A.get(1, 0));
        assertEquals(4, A.get(1, 1));
    }
    
    @Test
    public void plus() {
        
        Matrix A = new Matrix(new double[][] {
                {0, 1},
                {2, 3}});
        
        Matrix B = new Matrix(new double[][] {
                {1, 1},
                {1, 1}});
        
        Matrix expected = new Matrix(new double[][] {
                {1, 2},
                {3, 4}});
            
        assertEquals(expected, A.plus(B));
    }
    
    @Test
    public void minus() {
        
        Matrix A = new Matrix(new double[][] {
                {0, 1},
                {2, 3}});
        
        Matrix B = new Matrix(new double[][] {
                {1, 1},
                {1, 1}});
        
        Matrix expected = new Matrix(new double[][] {
                {-1, 0},
                { 1, 2}});
            
        assertEquals(expected, A.minus(B));
    }
    
    @Test
    public void matrixMultiplication() {
        
        Matrix A = new Matrix(new double[][] {
                {1, 2},
                {3, 4}});
        
        Matrix B = new Matrix(new double[][] {
                { 1, 2, 3},
                { 4, 5, 6}});
        
        Matrix expected = new Matrix(new double[][] {
                { 9, 12, 15},
                {19, 26, 33}});
        
        assertEquals(expected, A.multiply(B));
    }
    
    @Test
    public void scalarMultiplication() {
        
        Matrix A = new Matrix(new double[][] {
                {0, 1},
                {2, 3}});
        
        Matrix expected = new Matrix(new double[][] {
                {0, 2},
                {4, 6}});
        
        assertEquals(expected, A.scale(2));
    }
    
    @Test
    public void vectorMultiplication() {
        
        Matrix A = new Matrix(new double[][] {
                {0, 1},
                {2, 3},
                {4, 5}});
        
        Vector x = new Vector(1, 2);
        
        Vector expected = new Vector(2, 8, 14);
        
        assertEquals(expected, A.multiply(x));
    }
    
    @Test
    public void rowVector() {
        
        Matrix A = new Matrix(new double[][] {
                {0, 1},
                {2, 3}});
        
        Vector expected = new Vector(2, 3);
        
        assertEquals(expected, A.getRowVector(1));
    }
    
    @Test
    public void columnVector() {
        
        Matrix A = new Matrix(new double[][] {
                {0, 1},
                {2, 3}});
        
        Vector expected = new Vector(1, 3);
        
        assertEquals(expected, A.getColumnVector(1));
    }
    
    @Test
    public void identity() {
        
        Matrix expected = new Matrix(new double[][] {
                {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1}});
        
        assertEquals(expected, MatrixFactory.createIdentity(3));
    }
    
    @Test
    public void diagonal() {
        
        Matrix expected = new Matrix(new double[][] {
                {1, 0, 0},
                {0, 2, 0},
                {0, 0, 3}});
        
        assertEquals(expected, MatrixFactory.createDiagonal(1, 2, 3));
    }
    
    @Test
    public void augment() {
        
        Matrix A = new Matrix(new double[][] {
                {0, 1},
                {2, 3},
                {4, 5}});
        
        Matrix B = new Matrix(new double[][] {
                {5, 9},
                {6, 8},
                {7, 7}});
        
        Matrix expected = new Matrix(new double[][] {
                {0, 1, 5, 9},
                {2, 3, 6, 8},
                {4, 5, 7, 7}});
        
        assertEquals(expected, A.augment(B));
    }
    
    @Test
    @Disabled
    public void format() {
        Matrix matrix = new Matrix(new double[][] {
                {2, 4, 15},
                {1, 200, 3},
                {2, 1, 3}});
        
        String expected = """
                  2,00   4,00  15,00
                  1,00 200,00   3,00
                  2,00   1,00   3,00""";
        
        assertEquals(expected, "\n" + matrix.toString());
    }
    
}
