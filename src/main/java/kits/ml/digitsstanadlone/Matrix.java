package kits.ml.digitsstanadlone;

import static java.util.stream.Collectors.joining;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Matrix {

    private final int nrRows;
    private final int nrCols;
    
    final double[][] values;

    public Matrix(int nrRows, int nrColumns) {
        this(new double[nrRows][nrColumns]);
    }
    
    public Matrix(double[] ... values) {
        
        if(values.length == 0) throw new IllegalArgumentException("No values provided");
        
        this.nrRows = values.length;
        this.nrCols = values[0].length;
        
        if(Stream.of(values).anyMatch(row -> row.length != nrCols)) throw new IllegalArgumentException("All rows must contain the same number of values");
        
        this.values = new double[nrRows][nrCols];
        
        for (int rowIndex=0;rowIndex<values.length;rowIndex++) {
            this.values[rowIndex] = values[rowIndex].clone();            
        }
    }
    
    public Matrix(Matrix a) {
        this(a.values);
    }

    public int getNrRows() {
        return nrRows;
    }

    public int getNrColumns() {
        return nrCols;
    }
    
    public void set(int rowIndex, int columnIndex, double value) {
        if(rowIndex >= nrRows || columnIndex >= nrCols) throw new IllegalArgumentException("Illegal index. rowIndex must be < " + nrRows + " columnIndex must be < " + nrCols);
        values[rowIndex][columnIndex] = value;
    }
    
    public double get(int rowIndex, int columnIndex) {
        if(rowIndex >= nrRows || columnIndex >= nrCols) throw new IllegalArgumentException("Illegal index. rowIndex must be < " + nrRows + " columnIndex must be < " + nrCols);
        return values[rowIndex][columnIndex];
    }
    
    public Vector getRowVector(int rowIndex) {
        if(rowIndex >= nrRows) throw new IllegalArgumentException("Illegal index. rowIndex must be < " + nrRows);
        // creating a new object with array copy is expensive, makes multiplications slow
        return new Vector(values[rowIndex]);
    }
    
    public Matrix plus(Matrix other) {
        return applyOperation(other, (a, b) -> a + b);
    }
    
    public Matrix minus(Matrix other) {
        return applyOperation(other, (a, b) -> a - b);
    }
    
    private Matrix applyOperation(Matrix other, BinaryOperator<Double> operator) {
        if(nrRows != other.nrRows || nrCols != other.nrCols) throw new IllegalArgumentException("Dimension mismatch: " + printDimenstions() + " vs " + other.printDimenstions());
        
        double[][] resultValues = new double[nrRows][nrCols];
        for(int rowIndex=0;rowIndex<nrRows;rowIndex++) {
            for(int columnIndex=0;columnIndex<nrCols;columnIndex++) {
                resultValues[rowIndex][columnIndex] = operator.apply(values[rowIndex][columnIndex], other.values[rowIndex][columnIndex]);
            }
        }
        
        return new Matrix(resultValues);
    }
    
    public void minusThis(Matrix other) {
        applyOperationOnThis(other, (a, b) -> a - b);
    }
    
    private void applyOperationOnThis(Matrix other, BinaryOperator<Double> operator) {
        if(nrRows != other.nrRows || nrCols != other.nrCols) throw new IllegalArgumentException("Dimension mismatch: " + printDimenstions() + " vs " + other.printDimenstions());
        
        for(int rowIndex=0;rowIndex<nrRows;rowIndex++) {
            for(int columnIndex=0;columnIndex<nrCols;columnIndex++) {
                values[rowIndex][columnIndex] = operator.apply(values[rowIndex][columnIndex], other.values[rowIndex][columnIndex]);
            }
        }
    }
    
    public Vector multiply(Vector x) {
        if(nrCols != x.length()) throw new IllegalArgumentException("Dimension mismatch: " + printDimenstions() + " vs " + x.length());
        
        Vector result = Vector.createZero(nrRows);
        for(int index=0;index<nrRows;index++) {
            result.set(index, getRowVector(index).scalarProduct(x));
        }
        
        return result;
    }
    
    public String printDimenstions() {
        return nrRows + " X " + nrCols;
    }
    
    public Matrix transpose() {
        double[][] resultValues = new double[nrCols][nrRows];
        for(int rowIndex=0;rowIndex<nrRows;rowIndex++) {
            for(int columnIndex=0;columnIndex<nrCols;columnIndex++) {
                resultValues[columnIndex][rowIndex] = values[rowIndex][columnIndex];
            }
        }
        return new Matrix(resultValues);
    }
    
    public void apply(BiFunction<Integer, Integer, Double> mapper) {
        for(int rowIndex=0;rowIndex<nrRows;rowIndex++) {
            for(int columnIndex=0;columnIndex<nrCols;columnIndex++) {
                values[rowIndex][columnIndex] = mapper.apply(rowIndex, columnIndex);
            }
        }
    }
    
    @Override
    public String toString() {
        return toString(2);
    }
    
    public String toString(int fractionDigits) {

        int[] rowIndexes = nrRows <= 6 ? IntStream.range(0,  nrRows).toArray() : new int[] {0, 1, 2, nrRows-3, nrRows-2, nrRows-1};
        int[] colIndexes = nrCols <= 6 ? IntStream.range(0,  nrCols).toArray() : new int[] {0, 1, 2, nrCols-3, nrCols-2, nrCols-1};
        
        double max = Double.MIN_VALUE;
        for(int rowIndex : rowIndexes) {
            for(int colIndex : colIndexes) {
                if(values[rowIndex][colIndex] > max) {
                    max = values[rowIndex][colIndex];
                }
            }
        }
        
        int maxDigits = (int)Math.log10(max) + 2;
        String formatPattern = "%" + (maxDigits + fractionDigits + 1) + "." + fractionDigits + "f";
        
        List<List<String>> stringValues = new ArrayList<>();
        for(int rowIndex : rowIndexes) {
            List<String> rowStringValues = new ArrayList<>();
            for(int colIndex : colIndexes) {
                rowStringValues.add(String.format(formatPattern, get(rowIndex, colIndex)));
                if(nrCols > 6 && colIndex == 2) {
                    rowStringValues.add("...");
                }
            }
            stringValues.add(rowStringValues);
            if(nrRows > 6 && rowIndex == 2) {
                stringValues.add(List.of("..."));
            }
        }
        
        return stringValues.stream().map(row -> String.join(" ", row)).collect(joining("\n"));
    }

}
