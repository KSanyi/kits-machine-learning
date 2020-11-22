package kits.ml.core.math.linalg;

import kits.ml.core.math.linalg.Decomposition.LU;

public class GaussEliminationCalculator {

    public static Matrix runGaussElimination(Matrix A, Matrix B) {
        
        if(A.getNrRows() != A.getNrColumns()) throw new IllegalArgumentException("Not a square matrix");
        
        if(A.getNrRows() != B.getNrRows()) throw new IllegalArgumentException("Rownum of A and B matrixes must match");
        
        int n = A.getNrColumns();
        
        Matrix AB = A.augment(B);
        
        for(int rowIndex=0;rowIndex<n-1;rowIndex++) {
            int pivotIndex = findPivotIndex(AB, rowIndex);
            AB.swapRows(rowIndex, pivotIndex);
            double pivot = AB.get(rowIndex, rowIndex);
            Vector pivotRow = AB.getRowVector(rowIndex);
            for(int columnIndex=rowIndex+1;columnIndex<n;columnIndex++) {
                Vector row = AB.getRowVector(columnIndex);
                row = row.minus(pivotRow.multiply(row.get(rowIndex) / pivot));
                AB.setRowVector(columnIndex, row);
            }
        }
        
        for(int rowIndex=n-1;rowIndex>=0;rowIndex--) {
            double pivot = AB.get(rowIndex, rowIndex);
            Vector pivotRow = AB.getRowVector(rowIndex);
            if(pivot == 0) {
                swapRowsUp(AB, rowIndex);
                pivot = AB.get(rowIndex, rowIndex);
                pivotRow = AB.getRowVector(rowIndex);
            }
            if(pivot != 1) {
                pivotRow = pivotRow.multiply(1 / pivot);
                AB.setRowVector(rowIndex, pivotRow);
                pivot = 1;
            }
            for(int columnIndex=rowIndex-1;columnIndex>=0;columnIndex--) {
                Vector row = AB.getRowVector(columnIndex);
                row = row.minus(pivotRow.multiply(row.get(rowIndex) / pivot));
                AB.setRowVector(columnIndex, row);
            }
        }
        
        return AB;
    }
    
    /**
     *  Scaled pivoting: in this approach, the algorithm selects the largest entry as the pivot element to
     *  prevent propagations of rounding errors.
     */
    private static int findPivotIndex(Matrix Ab, int rowIndex) {
        double max = Ab.get(rowIndex, rowIndex);
        int rowIndexForMax = rowIndex;
        for(int i=rowIndex+1;i<Ab.getNrRows();i++) {
            double candidate = Math.abs(Ab.get(i, rowIndex));
            if(candidate > max) {
                max = candidate; 
                rowIndexForMax = i;
            }
        }
        if(max == 0) {
            throw new IllegalArgumentException("No solution");    
        } else {
            return rowIndexForMax;
        }
        
    }

    public static Vector solveEquation(Matrix A, Vector b) {
        
        Matrix AB = runGaussElimination(A, b.asMatrix());
        
        return AB.getColumnVector(A.getNrColumns());
    }
    
    public static Matrix calculateInverse(Matrix A) {
        
        Matrix AB = runGaussElimination(A, Matrix.createIdentity(A.getNrColumns()));
        
        return AB.getSubMatrix(0, A.getNrRows(), A.getNrRows(), 2 * A.getNrRows());
    }
    
    private static void swapRowsUp(Matrix Ab, int rowIndex) {
        for(int i=rowIndex-1;i>=0;i--) {
            if(Ab.get(i, rowIndex) != 0) {
                Ab.swapRows(i, rowIndex);
                return;
            }
        }
        throw new IllegalArgumentException("No solution");
        
    }
    
    /**
     * Create a Doolittle decomposition: the main diagonal of L is composed solely of 1s.
     * The decomposition includes a permutation matrix as well: PA = LU or A = P*LU (as PP* = P*P = I)
     * 
     * With a sequence of elementary row operations we modify the matrix until we reach U.
     * During this we capture these row operations in L. (Elementary row operations can be achieved by multiplying the matrix from left with an elementary matrix. Multiples of these elementary matrixes is a lower triangular matrix)
     * Sometimes we need row swaps that we record in P. (Row swaps can be seen as multiplying from left with a row swap elementary matrix. Multiples of these elementary matrixes is a permutation matrix)
     */
    public static LU createLUDecomposition(Matrix A) {
        
        if(A.getNrRows() != A.getNrColumns()) throw new IllegalArgumentException("Not a square matrix");
        
        int n = A.getNrColumns();
        
        Matrix L = Matrix.createZero(n); // we don't want row swaps to mess up the diagonal, so we add the diagonal in the end
        Matrix U = new Matrix(A);
        Matrix P = Matrix.createIdentity(n);
        
        for(int rowIndex=0;rowIndex<n-1;rowIndex++) {
            int pivotIndex = findPivotIndex(U, rowIndex);
            U.swapRows(rowIndex, pivotIndex);
            P.swapRows(rowIndex, pivotIndex);
            L.swapRows(rowIndex, pivotIndex);
            double pivot = U.get(rowIndex, rowIndex);
            Vector pivotRow = U.getRowVector(rowIndex);
            for(int subRowIndex=rowIndex+1;subRowIndex<n;subRowIndex++) {
                Vector row = U.getRowVector(subRowIndex);
                double quotient = row.get(rowIndex) / pivot;
                L.set(subRowIndex, rowIndex, quotient);
                row = row.minus(pivotRow.multiply(quotient));
                U.setRowVector(subRowIndex, row);
            }
        }
        
        L = L.plus(Matrix.createIdentity(n));
        
        return new LU(L, U, P);
    }
    
}
