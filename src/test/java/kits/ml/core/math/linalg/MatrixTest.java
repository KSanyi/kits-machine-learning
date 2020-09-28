package kits.ml.core.math.linalg;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class MatrixTest {

    @Test
    public void test() {
        
        Matrix A = new Matrix(2, 2);
        A.set(1, 1, 3.0);
        
        Matrix expected = new Matrix(
            new double[] {0, 0},
            new double[] {0, 3});
        
       assertEquals(expected, A);
    }
    
    @Test
    public void plus() {
        
        Matrix A = new Matrix(
                new double[] {0, 1},
                new double[] {2, 3});
        
        Matrix B = new Matrix(
                new double[] {1, 1},
                new double[] {1, 1});
        
        Matrix expected = new Matrix(
                new double[] {1, 2},
                new double[] {3, 4});
            
        assertEquals(expected, A.plus(B));
    }
    
    @Test
    public void minus() {
        
        Matrix A = new Matrix(
                new double[] {0, 1},
                new double[] {2, 3});
        
        Matrix B = new Matrix(
                new double[] {1, 1},
                new double[] {1, 1});
        
        Matrix expected = new Matrix(
                new double[] {-1, 0},
                new double[] { 1, 2});
            
        assertEquals(expected, A.minus(B));
    }
    
    @Test
    public void multiplication() {
        
        Matrix A = new Matrix(
                new double[] {1, 2},
                new double[] {3, 4});
        
        Matrix B = new Matrix(
                new double[] { 1, 2, 3},
                new double[] { 4, 5, 6});
        
        Matrix expected = new Matrix(
                new double[] { 9, 12, 15},
                new double[] {19, 26, 33});
        
        Assertions.assertEquals(expected, A.multiply(B));
    }
    
    @Test
    public void multiply() {
        
        Matrix A = new Matrix(
                new double[] {0, 1},
                new double[] {2, 3});
        
        Matrix expected = new Matrix(
                new double[] {0, 2},
                new double[] {4, 6});
        
        Assertions.assertEquals(expected, A.multiply(2));
    }
    
    @Test
    public void multiplyWithVector() {
        
        Matrix A = new Matrix(
                new double[] {0, 1},
                new double[] {2, 3},
                new double[] {4, 5});
        
        Vector x = new Vector(1, 2);
        
        Vector expected = new Vector(2, 8, 14);
        
        Assertions.assertEquals(expected, A.multiply(x));
    }
    
    @Test
    public void rowVector() {
        
        Matrix A = new Matrix(
                new double[] {0, 1},
                new double[] {2, 3});
        
        Vector expected = new Vector(2, 3);
        
        Assertions.assertEquals(expected, A.getRowVector(1));
    }
    
    @Test
    public void columnVector() {
        
        Matrix A = new Matrix(
                new double[] {0, 1},
                new double[] {2, 3});
        
        Vector expected = new Vector(1, 3);
        
        Assertions.assertEquals(expected, A.getColumnVector(1));
    }
    
    @Test
    public void identity() {
        
        Matrix expected = new Matrix(
                new double[] {1, 0, 0},
                new double[] {0, 1, 0},
                new double[] {0, 0, 1});
        
        Assertions.assertEquals(expected, Matrix.createIdentity(3));
    }
    
    @Test
    public void diagonal() {
        
        Matrix expected = new Matrix(
                new double[] {1, 0, 0},
                new double[] {0, 2, 0},
                new double[] {0, 0, 3});
        
        Assertions.assertEquals(expected, Matrix.createDiagonal(1, 2, 3));
    }
    
    @Test
    public void appendMatrix() {
        
        Matrix A = new Matrix(
                new double[] {0, 1},
                new double[] {2, 3},
                new double[] {4, 5});
        
        Matrix B = new Matrix(
                new double[] {5, 9},
                new double[] {6, 8},
                new double[] {7, 7});
        
        Matrix expected = new Matrix(
                new double[] {0, 1, 5, 9},
                new double[] {2, 3, 6, 8},
                new double[] {4, 5, 7, 7});
        
        Assertions.assertEquals(expected, A.appendMatrix(B));
    }
    
    @Test
    public void format() {
        Matrix matrix = new Matrix(
                new double[] {2, 4, 15},
                new double[] {1, 200, 3},
                new double[] {2, 1, 3});
        
        String expected = """
                  2,00   4,00  15,00
                  1,00 200,00   3,00
                  2,00   1,00   3,00""";
        
        //Assertions.assertEquals(expected, "\n" + matrix.toString());
    }
    
}
