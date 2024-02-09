package kits.ml.core.math.linalg;

import java.util.Random;

import org.junit.jupiter.api.Test;

import kits.ml.util.StopWatch;

public class CholeskyDecompositionCalculatorTest {

    //@Test
    public void test() {
        Random random = new Random();
        Matrix L = RandomUtils.generateRandomLowerTriangularMatrix(10, () -> random.nextDouble(100));
        
        Matrix A = StopWatch.timed(() -> L.multiply(L.transpose()), "L x L");
        
        printSample(L);
        printSample(A);
        
        Matrix cL = CholeskyDecompositionCalculator.calculate(A);
        
        System.out.println();
        printSample(cL);
        
        int n = 10000;
        Matrix X = MatrixFactory.createDiagonal(n, 99);
        Matrix C = X.plus(MatrixFactory.createOnes(n));
        
        printSample(C);
        
        Matrix x = StopWatch.timed(() -> CholeskyDecompositionCalculator.calculateAlex(C), "Cholesky");
        
        printSample(x);
    }

    @Test
    public void testDegenarate() {
        Random random = new Random();
        Matrix L = RandomUtils.generateRandomMatrix(10, 5, () -> random.nextDouble(100));
        
        Matrix A = StopWatch.timed(() -> L.multiply(L.transpose()), "L x L");
        
        System.out.println(L);
        printSample(L);
        System.out.println("A");
        printSample(A);
        
        Matrix cL = CholeskyDecompositionCalculator.calculate(A);
        
        System.out.println();
        printSample(cL);
        
        printSample(cL.multiply(cL));
    }
    
    private static void printSample(Matrix M) {
        System.out.println(M);//.getSubMatrix(0, 5, 0, 5));
        System.out.println();
    }
    
}
