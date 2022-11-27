package kits.ml.core.math.linalg;

import java.util.Random;

import org.junit.jupiter.api.Test;

import kits.ml.util.StopWatch;

public class CholeskyDecompositionCalculatorTest {

    @Test
    public void test() {
        Random random = new Random();
        Matrix L = RandomUtils.generateRandomLowerTriangularMatrix(1000, () -> random.nextDouble(100));
        
        Matrix A = StopWatch.timed(() -> L.multiply(L.transpose()), "L x L");
        
        System.out.println(L.getSubMatrix(0, 5, 0, 5));
        System.out.println();
        System.out.println(A.getSubMatrix(0, 5, 0, 5));
        
        Matrix cL = CholeskyDecompositionCalculator.calculate(A);
        
        System.out.println();
        System.out.println(cL.getSubMatrix(0, 5, 0, 5));
        
        int n = 10000;
        
        Matrix X = MatrixFactory.createDiagonal(n, 99);
        X = X.plus(MatrixFactory.createOnes(n));
    }
    
}
