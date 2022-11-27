package kits.ml.core.math.linalg;

import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

public class MatrixFactory {

    public static Matrix fromColumnVectors(List<Vector> coulumnVectors) {
        return fromColumnVectors(coulumnVectors.toArray(new Vector[0]));
    }
    
    public static Matrix fromColumnVectors(Vector ... coulumnVectors) {
        
        if(coulumnVectors.length == 0) throw new IllegalArgumentException("No values provided");
        
        int nrColumns = coulumnVectors.length;
        int nrRows = coulumnVectors[0].length();
        
        if(Stream.of(coulumnVectors).anyMatch(column -> column.length() != nrRows)) throw new IllegalArgumentException("All columns must contain the same number of values");
        
        double[][] values = new double[nrRows][nrColumns];
        
        for (int rowIndex=0;rowIndex<values.length;rowIndex++) {
            values[rowIndex] = new double[nrColumns];
            for(int columnIndex=0;columnIndex<nrColumns;columnIndex++) {
                values[rowIndex][columnIndex] = coulumnVectors[columnIndex].get(rowIndex);
            }
        }
        
        return new Matrix(values);
    }
    
    public static Matrix createZero(int n) {
        return new Matrix(n, n);
    }
    
    public static Matrix createDiagonal(double ... values) {
        int n = values.length;
        Matrix matrix = new Matrix(n, n);
        for(int index=0;index<n;index++) {
            matrix.set(index, index, values[index]);
        }
        return matrix;
    }
    
    public static Matrix createDiagonal(int n, double value) {
        return createDiagonal(DoubleStream.generate(() -> value).limit(n).toArray());
    }
    
    public static Matrix createIdentity(int n) {
        return createDiagonal(n, 1);
    }
    
    public static Matrix createWithSameValue(int n, double value) {
        double[][] data = new double[n][n];
        for(int i=0;i<n;i++) {
            for(int j=0;j<n;j++) {
                data[i][j] = value;
            }
        }
        return new Matrix(data);
    }
    
    public static Matrix createOnes(int n) {
        return createWithSameValue(n, 1);
    }
    
}
