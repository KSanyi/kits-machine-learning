package kits.ml.core.math;

public class GaussEliminationCalculator {

    public static Matrix runGaussElimination(Matrix A, Matrix B) {
        
        if(A.getNrRows() != A.getNrColumns()) throw new IllegalArgumentException("Not a square matrix");
        
        if(A.getNrRows() != B.getNrRows()) throw new IllegalArgumentException("Rownum of A and B matrixes must match");
        
        int n = A.getNrColumns();
        
        Matrix AB = A.appendMatrix(B);
        
        for(int rowIndex=0;rowIndex<n-1;rowIndex++) {
            double pivot = AB.get(rowIndex, rowIndex);
            Vector pivotRow = AB.getRowVector(rowIndex);
            if(pivot == 0) {
                swapRowsDown(AB, rowIndex);
                pivot = AB.get(rowIndex, rowIndex);
                pivotRow = AB.getRowVector(rowIndex);
            }
            for(int columnIndex=rowIndex+1;columnIndex<n;columnIndex++) {
                Vector row = AB.getRowVector(columnIndex);
                row = row.minus(pivotRow.multiply(row.get(rowIndex) / pivot));
                AB.setRowVector(columnIndex, row);
            }
            //System.out.println(Ab);
            //System.out.println();
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
    
    public static Vector solveEquations(Matrix A, Vector b) {
        
        Matrix AB = runGaussElimination(A, b.asMatrix());
        
        return AB.getColumnVector(A.getNrColumns());
    }
    
    public static Matrix calculateInverse(Matrix A) {
        
        Matrix AB = runGaussElimination(A, Matrix.createIdentity(A.getNrColumns()));
        
        return AB.getSubMatrix(0, A.getNrRows(), A.getNrRows(), 2 * A.getNrRows());
    }
    
    private static void swapRowsDown(Matrix Ab, int rowIndex) {
        for(int i=rowIndex+1;i<Ab.getNrRows();i++) {
            if(Ab.get(i, rowIndex) != 0) {
                Ab.swapRows(i, rowIndex);
                return;
            }
        }
        throw new IllegalArgumentException("No solution");
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
    
}
