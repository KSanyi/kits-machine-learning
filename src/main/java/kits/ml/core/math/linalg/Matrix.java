package kits.ml.core.math.linalg;

import static java.util.stream.Collectors.joining;
import static java.util.stream.Collectors.toList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Matrix {

    private static final double EPSILON = 0.001;
    
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
    
    public void setRowVector(int rowIndex, Vector row) {
        for(int columnIndex=0;columnIndex<nrCols;columnIndex++) {
            set(rowIndex, columnIndex, row.get(columnIndex));
        }
    }
    
    public Vector getColumnVector(int columnIndex) {
        if(columnIndex >= nrCols) throw new IllegalArgumentException("Illegal index. columnIndex must be < " + nrCols);
        double[] columnVectorValues = new double[nrRows];
        for(int rowIndex=0;rowIndex<nrRows;rowIndex++) {
            columnVectorValues[rowIndex] = values[rowIndex][columnIndex];
        }
        // creating a new object with array copy is expensive, makes multiplications slow
        return new Vector(columnVectorValues);
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
    
    public Matrix scale(double lambda) {
        double[][] resultValues = new double[nrRows][nrCols];
        for(int rowIndex=0;rowIndex<nrRows;rowIndex++) {
            for(int columnIndex=0;columnIndex<nrCols;columnIndex++) {
                resultValues[rowIndex][columnIndex] = lambda * values[rowIndex][columnIndex];
            }
        }
        return new Matrix(resultValues);
    }
    
    // just for educational purposes, this is 3 times slower than the below one
    public Matrix multiplySlow(Matrix other) {
        if(nrCols != other.nrRows) throw new IllegalArgumentException("Dimension mismatch: " + printDimenstions() + " vs " + other.printDimenstions());
        
        double[][] resultValues = new double[nrRows][other.nrCols];
        for(int rowIndex=0;rowIndex<nrRows;rowIndex++) {
            for(int columnIndex=0;columnIndex<other.nrCols;columnIndex++) {
                resultValues[rowIndex][columnIndex] = getRowVector(rowIndex).scalarProduct(other.getColumnVector(columnIndex));
            }
        }
        return new Matrix(resultValues);
    }
    
    public Matrix multiply(Matrix other) {
        if(nrCols != other.nrRows) throw new IllegalArgumentException("Dimension mismatch: " + printDimenstions() + " vs " + other.printDimenstions());
        
        double[][] product = new double[nrRows][other.nrCols];
        
        for (int i = 0; i < nrRows; i++) {
            for (int j = 0; j < other.nrRows; j++) {
                double elemIJ = values[i][j];
                for(int k = 0; k < other.nrCols; k++) {
                    product[i][k] += elemIJ * other.values[j][k];
                }
            }
        }
        
        return new Matrix(product);
    }
    
    public Vector multiply(Vector x) {
        if(nrCols != x.length()) throw new IllegalArgumentException("Dimension mismatch: " + printDimenstions() + " vs " + x.length());
        
        Vector result = new Vector(nrRows);
        for(int index=0;index<nrRows;index++) {
            result.set(index, getRowVector(index).scalarProduct(x));
        }
        
        return result;
    }
    
    public Matrix augment(Vector b) {
        return augment(b.asMatrix());
    }
    
    public Matrix augment(Matrix other) {
        if(other.nrRows != nrRows) throw new IllegalArgumentException("Rownum of matrixes must match");
        double[][] resultValues = new double[nrRows][nrCols + other.nrCols];
        for(int rowIndex=0;rowIndex<nrRows;rowIndex++) {
            for(int columnIndex=0;columnIndex<nrCols;columnIndex++) {
                resultValues[rowIndex][columnIndex] = values[rowIndex][columnIndex];
            }
            for(int columnIndex=0;columnIndex<other.nrCols;columnIndex++) {
                resultValues[rowIndex][nrCols+columnIndex] = other.values[rowIndex][columnIndex];
            }
        }
        return new Matrix(resultValues);
    }
    
    public void swapRows(int i, int j) {
        if(i >= nrRows || j >= nrRows) throw new IllegalArgumentException("Illegal index. index must be < " + nrRows);
        
        if(i == j) return;
        
        Vector rowVector_i = getRowVector(i);
        Vector rowVector_j = getRowVector(j);
        setRowVector(i, rowVector_j);
        setRowVector(j, rowVector_i);
    }
    
    public String printDimenstions() {
        return nrRows + " X " + nrCols;
    }
    
    public DoubleStream allValuesStream() {
        return Arrays.stream(values)
                .map(DoubleStream::of)
                .reduce(DoubleStream::concat)
                .get();
    }
    
    public Matrix getSubMatrix(int rowStart, int rowEnd, int columnStart, int columnEnd) {
        double[][] values = new double[rowEnd - rowStart][columnEnd - columnStart];
        for(int rowIndex=rowStart;rowIndex<rowEnd;rowIndex++) {
            for(int columnIndex=columnStart;columnIndex<columnEnd;columnIndex++) {
                values[rowIndex - rowStart][columnIndex - columnStart] = this.values[rowIndex][columnIndex];
            }   
        }
        return new Matrix(values);
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
    
    public Matrix map(BiFunction<Integer, Integer, Double> mapper) {
        double[][] resultValues = new double[nrRows][nrCols];
        for(int rowIndex=0;rowIndex<nrRows;rowIndex++) {
            for(int columnIndex=0;columnIndex<nrCols;columnIndex++) {
                resultValues[rowIndex][columnIndex] = mapper.apply(rowIndex, columnIndex);
            }
        }
        return new Matrix(resultValues);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null) return false;
        if (getClass() != obj.getClass()) return false;
        Matrix other = (Matrix) obj;
        List<Double> allValues = allValuesStream().mapToObj(d -> d).collect(toList());
        List<Double> allOtherValues = other.allValuesStream().mapToObj(d -> d).collect(toList());
        return IntStream.range(0, allValues.size()).allMatch(index -> Math.abs(allValues.get(index) - allOtherValues.get(index)) < EPSILON);
    }
    
    @Override
    public int hashCode() {
        return toString(2).hashCode();
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
