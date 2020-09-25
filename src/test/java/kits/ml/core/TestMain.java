package kits.ml.core;

import kits.ml.core.math.GaussEliminationCalculator;
import kits.ml.core.math.Matrix;
import kits.ml.core.math.MatrixUtils;
import kits.ml.core.math.Vector;
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
                Vector result = GaussEliminationCalculator.runGaussElimination(A, b);
                //System.out.println(result);
                //System.out.println(x);
            }
        });
        
    }

}
