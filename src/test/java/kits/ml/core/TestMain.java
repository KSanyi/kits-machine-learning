package kits.ml.core;

import kits.ml.core.math.linalg.GaussEliminationCalculator;
import kits.ml.core.math.linalg.Matrix;
import kits.ml.core.math.linalg.MatrixUtils;
import kits.ml.core.math.linalg.Vector;
import kits.ml.util.StopWatch;

public class TestMain {

    public static void main(String[] args) {
        
        Matrix A = MatrixUtils.generateRandomIntMatrix(200, 200, 100);
        
        Vector x = MatrixUtils.generateRandomIntVector(200, 100);

        Vector b = A.multiply(x);

        System.out.println("x");
        //System.out.println(x);
        
        System.out.println("A");
        //System.out.println(A);

        System.out.println("b");
        //System.out.println(b);
        
        StopWatch.timed(() -> {
            for(int i=0;i<1000;i++) {
                Vector result = GaussEliminationCalculator.solveEquations(A, b);
                //System.out.println(result);
                //System.out.println(x);
            }
        });
        
    }

}
