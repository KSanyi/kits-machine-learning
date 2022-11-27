package kits.ml.core.math.linalg;

import java.util.Random;
import java.util.function.Supplier;

public class RandomUtils {

    public static Matrix generateRandomIntMatrix(int nrRows, int nrColumns, int maxNumber) {
        
        Random random = new Random();
        
        return generateRandomMatrix(nrRows, nrColumns, () -> (double)random.nextInt(maxNumber));
    }
    
    public static Matrix generateRandomMatrix(int nrRows, int nrColumns, Supplier<Double> elementCreator) {
        
        Matrix matrix = new Matrix(nrRows, nrColumns);
        for(int rowIndex=0;rowIndex<nrRows;rowIndex++) {
            for(int columnIndex=0;columnIndex<nrColumns;columnIndex++) {
                matrix.set(rowIndex, columnIndex, elementCreator.get());
            }
        }
        
        return matrix;
    }
    
    public static Matrix generateRandomLowerTriangularMatrix(int nrRows, Supplier<Double> elementCreator) {
        
        Matrix matrix = new Matrix(nrRows, nrRows);
        for(int rowIndex=0;rowIndex<nrRows;rowIndex++) {
            for(int columnIndex=0;columnIndex<=rowIndex;columnIndex++) {
                matrix.set(rowIndex, columnIndex, elementCreator.get());
            }
        }
        
        return matrix;
    }
    
    public static Vector generateRandomVector(int length, Supplier<Double> elementCreator) {
        
        Vector vector = new Vector(length);
        for(int index=0;index<length;index++) {
            vector.set(index, elementCreator.get());
        }
        
        return vector;
    }
    
    public static Vector generateRandomIntVector(int length, int maxNumber) {
        
        Random random = new Random();
        
        return generateRandomVector(length, () -> (double)random.nextInt(maxNumber));
    }
    
}
