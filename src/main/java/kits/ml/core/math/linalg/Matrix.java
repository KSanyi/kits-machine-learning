package kits.ml.core.math.linalg;

import static java.util.stream.Collectors.joining;
import static java.util.stream.Collectors.toList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BinaryOperator;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Matrix {

    private static final double EPSILON = 0.001;
    
    private final int nrRows;
    private final int nrColumns;
    
    private final double[][] values;

    public Matrix(int nrRows, int nrColumns) {
        this(new double[nrRows][nrColumns]);
    }
    
    public Matrix(double[] ... values) {
        
        if(values.length == 0) throw new IllegalArgumentException("No values provided");
        
        this.nrRows = values.length;
        this.nrColumns = values[0].length;
        
        if(Stream.of(values).anyMatch(row -> row.length != nrColumns)) throw new IllegalArgumentException("All rows must contain the same number of values");
        
        this.values = new double[nrRows][nrColumns];
        
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
        return nrColumns;
    }
    
    public void set(int rowIndex, int columnIndex, double value) {
        if(rowIndex >= nrRows || columnIndex >= nrColumns) throw new IllegalArgumentException("Illegal index. rowIndex must be < " + nrRows + " columnIndex must be < " + nrColumns);
        values[rowIndex][columnIndex] = value;
    }
    
    public double get(int rowIndex, int columnIndex) {
        if(rowIndex >= nrRows || columnIndex >= nrColumns) throw new IllegalArgumentException("Illegal index. rowIndex must be < " + nrRows + " columnIndex must be < " + nrColumns);
        return values[rowIndex][columnIndex];
    }
    
    public Vector getRowVector(int rowIndex) {
        if(rowIndex >= nrRows) throw new IllegalArgumentException("Illegal index. rowIndex must be < " + nrRows);
        return new Vector(values[rowIndex]);
    }
    
    public void setRowVector(int rowIndex, Vector row) {
        for(int columnIndex=0;columnIndex<nrColumns;columnIndex++) {
            set(rowIndex, columnIndex, row.get(columnIndex));
        }
    }
    
    public Vector getColumnVector(int columnIndex) {
        if(columnIndex >= nrColumns) throw new IllegalArgumentException("Illegal index. columnIndex must be < " + nrColumns);
        double[] columnVectorValues = new double[nrRows];
        for(int rowIndex=0;rowIndex<nrRows;rowIndex++) {
            columnVectorValues[rowIndex] = values[rowIndex][columnIndex];
        }
        return new Vector(columnVectorValues);
    }
    
    public Matrix plus(Matrix other) {
        return applyOperation(other, (a, b) -> a + b);
    }
    
    public Matrix minus(Matrix other) {
        return applyOperation(other, (a, b) -> a - b);
    }
    
    private Matrix applyOperation(Matrix other, BinaryOperator<Double> operator) {
        if(nrRows != other.nrRows || nrColumns != other.nrColumns) throw new IllegalArgumentException("Dimension mismatch: " + printDimenstions() + " vs " + other.printDimenstions());
        
        double[][] resultValues = new double[nrRows][nrColumns];
        for(int rowIndex=0;rowIndex<nrRows;rowIndex++) {
            for(int columnIndex=0;columnIndex<nrColumns;columnIndex++) {
                resultValues[rowIndex][columnIndex] = operator.apply(values[rowIndex][columnIndex], other.values[rowIndex][columnIndex]);
            }
        }
        
        return new Matrix(resultValues);
    }
    
    public Matrix multiply(double lambda) {
        double[][] resultValues = new double[nrRows][nrColumns];
        for(int rowIndex=0;rowIndex<nrRows;rowIndex++) {
            for(int columnIndex=0;columnIndex<nrColumns;columnIndex++) {
                resultValues[rowIndex][columnIndex] = lambda * values[rowIndex][columnIndex];
            }
        }
        return new Matrix(resultValues);
    }
    
    public Matrix multiply(Matrix other) {
        if(nrColumns != other.nrRows) throw new IllegalArgumentException("Dimension mismatch: " + printDimenstions() + " vs " + other.printDimenstions());
        
        double[][] resultValues = new double[nrRows][other.nrColumns];
        for(int rowIndex=0;rowIndex<nrRows;rowIndex++) {
            for(int columnIndex=0;columnIndex<other.nrColumns;columnIndex++) {
                resultValues[rowIndex][columnIndex] = getRowVector(rowIndex).scalarProduct(other.getColumnVector(columnIndex));
            }
        }
        return new Matrix(resultValues);
    }
    
    public Vector multiply(Vector x) {
        if(nrColumns != x.length()) throw new IllegalArgumentException("Dimension mismatch: " + printDimenstions() + " vs " + x.length());
        
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
        double[][] resultValues = new double[nrRows][nrColumns + other.nrColumns];
        for(int rowIndex=0;rowIndex<nrRows;rowIndex++) {
            for(int columnIndex=0;columnIndex<nrColumns;columnIndex++) {
                resultValues[rowIndex][columnIndex] = values[rowIndex][columnIndex];
            }
            for(int columnIndex=0;columnIndex<other.nrColumns;columnIndex++) {
                resultValues[rowIndex][nrColumns+columnIndex] = other.values[rowIndex][columnIndex];
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
        return nrRows + " X " + nrColumns;
    }
    
    private double findMaxValue() {
        return allValuesStream().max().getAsDouble();
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
        double[][] resultValues = new double[nrColumns][nrRows];
        for(int rowIndex=0;rowIndex<nrRows;rowIndex++) {
            for(int columnIndex=0;columnIndex<nrColumns;columnIndex++) {
                resultValues[rowIndex][columnIndex] = values[columnIndex][rowIndex];
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

        int maxDigits = (int)Math.log10(findMaxValue()) + 2;
        String formatPattern = "%" + (maxDigits + fractionDigits + 1) + "." + fractionDigits + "f";
        
        List<List<String>> stringValues = new ArrayList<>();
        for(int rowIndex=0;rowIndex<nrRows;rowIndex++) {
            List<String> rowStringValues = new ArrayList<>();
            for(int columnIndex=0;columnIndex<nrColumns;columnIndex++) {
                rowStringValues.add(String.format(formatPattern, values[rowIndex][columnIndex]));
            }
            stringValues.add(rowStringValues);
        }
        
        return stringValues.stream().map(row -> String.join(" ", row)).collect(joining("\n"));
    }
    
    // factory methods
    
    public static Matrix createIdentity(int n) {
        return createDiagonal(DoubleStream.generate(() -> 1.0).limit(n).toArray());
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

}
