package kits.ml.core.math.linalg;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class VectorTest {

    @Test
    public void addition() {
        Vector vector1 = new Vector(0, 1);
        Vector vector2 = new Vector(1, 2);
        
        assertEquals(new Vector(1, 3), vector1.plus(vector2));
    }
    
    @Test
    public void subtraction() {
        Vector vector1 = new Vector(0, 5);
        Vector vector2 = new Vector(1, 2);
        
        assertEquals(new Vector(-1, 3), vector1.minus(vector2));
    }
    
    @Test
    public void multiplication() {
        Vector vector1 = new Vector(1, 2);
        Vector vector2 = new Vector(3, 4, 5);
        
        Matrix expectedResult = new Matrix(
                new double[] {3, 4,  5},
                new double[] {6, 8, 10});
        
        assertEquals(expectedResult, vector1.multiply(vector2));
    }
    
    @Test
    public void multiply() {
        Vector vector = new Vector(1, 2);
        
        assertEquals(new Vector(2, 4), vector.scale(2));
    }
    
    @Test
    public void scalarProduct() {
        Vector vector1 = new Vector(1, 2, 3);
        Vector vector2 = new Vector(4, 5, 6);
        
        assertEquals(32, vector1.scalarProduct(vector2));
    }
    
    @Test
    public void pseudoScalarProduct() {
        Vector vector1 = new Vector(1, 2, 3, 6, 6, 6);
        Vector vector2 = new Vector(4, 5, 6);
        
        assertEquals(32, vector1.pseudoScalarProduct(vector2));
    }
    
    @Test
    public void norm() {
        assertEquals(0, new Vector(0).norm());
        assertEquals(3, new Vector(-3.0).norm());
        assertEquals(5, new Vector(1, 2, 2, 4).norm());
    }
    
}
