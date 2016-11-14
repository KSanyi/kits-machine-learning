package kits.ml.coursera;

import java.util.stream.DoubleStream;

import Jama.Matrix;
import kits.ml.core.math.MLMath;

public class Exercise4 {

    static final Matrix X = new Matrix(new double[][] {
        {2.524413, -2.270407, 1.970960, -1.632063, 1.260501, -0.863710, 0.449632, -0.026554, -0.397055,  0.812717},
        {2.727892, -2.876773, 2.968075, -2.999971, 2.971822, -2.884192, 2.738836, -2.538661,  2.287675, -1.990902},
        {0.423360, -0.838246, 1.236355, -1.609719, 1.950864, -2.252962, 2.509967, -2.716735,  2.869128, -2.964095}});
        
         
    public static void main(String[] args) {
        //task1();
        //System.out.println();
        //task2();
        //System.out.println();
        task3();
    }
    
    
    private static void task3() {
        System.out.println("Task 3");
        Matrix XGrad = MLMath.sigmoidGradient(X);
        DoubleStream.of(XGrad.getColumnPackedCopy()).forEach(x -> System.out.format("%.5f ", x));
        System.out.println();
    }
}
