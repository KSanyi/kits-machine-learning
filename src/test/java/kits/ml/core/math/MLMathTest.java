package kits.ml.core.math;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import kits.ml.core.math.linalg.Vector;

class MLMathTest {

    private final double TOLERANCE = 0.001;
    
    @Test
    void test() {
        Vector result = MLMath.softMax(new Vector(new double[] {1, 2, 3, 4, 5}));
        
        assertEquals(0.011, result.get(0), TOLERANCE);
        assertEquals(0.031, result.get(1), TOLERANCE);
        assertEquals(0.086, result.get(2), TOLERANCE);
        assertEquals(0.234, result.get(3), TOLERANCE);
        assertEquals(0.636, result.get(4), TOLERANCE);
    }

}
