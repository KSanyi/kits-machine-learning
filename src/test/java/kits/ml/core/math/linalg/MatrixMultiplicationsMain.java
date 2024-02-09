package kits.ml.core.math.linalg;

public class MatrixMultiplicationsMain {

    public static void main(String[] args) {
        
        Matrix A = new Matrix(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}});

        // elementary row manipulation matrixes
        
        Matrix E1 = new Matrix(new double[][] {
                {1, 0, 0},
                {1, 1, 0},
                {0, 0, 1}});
        
        System.out.println(E1.multiply(A)); // row2 := row1 + row2
        System.out.println();
        
        Matrix E2 = new Matrix(new double[][] {
                {1, 0, 0},
                {1, 1, 0},
                {1, 0, 1}});
        
        System.out.println(E2.multiply(A)); // row2 := row1 + row2, row3 := row1 + row3
        System.out.println();
        
        Matrix E3 = new Matrix(new double[][] {
                {1, 1, 0},
                {0, 1, 0},
                {0, 0, 1}});
        
        System.out.println(E3.multiply(A)); // row1 := row1 + row2
        System.out.println();
        
        // permutation matrixes
        
        Matrix P1 = new Matrix(new double[][] {
                {0, 0, 1},
                {0, 1, 0},
                {1, 0, 0}});
        
        System.out.println(P1.multiply(A)); // sap row1 and row3
        System.out.println();
    }

}
