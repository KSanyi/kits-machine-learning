package kits.ml.core.math.linalg;

import static java.util.stream.Collectors.joining;

import java.util.Arrays;
import java.util.function.Function;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class Vector {

    private static final double EPSILON = 0.001;
    
    private final int length;
    
    private final double[] values;

    public Vector(int length) {
        this(new double[length]);
    }
    
    public Vector(double ... values) {
        this.values = values.clone();
        length = values.length;
    }
    
    public Vector(Vector other) {
        this.values = other.values.clone();
        length = values.length;
    }
    
    public int length() {
        return length;
    }
    
    public double norm() {
        return Math.sqrt(this.scalarProduct(this));
    }

    public void set(int index, double value) {
        if(index >= length) throw new IllegalArgumentException("Illegal index. Index must be < " + length);
        values[index] = value;
    }
    
    public double get(int index) {
        if(index >= length) throw new IllegalArgumentException("Illegal index. Index must be < " + length);
        return values[index];
    }
    
    public Vector plus(Vector other) {
        if(length != other.length) throw new IllegalArgumentException("Dimension mismatch: " + length + " vs " + other.length);
        
        double[] resultValues = new double[length];
        for(int i=0;i<length;i++) {
            resultValues[i] = values[i] + other.values[i];
        }
        
        return new Vector(resultValues);
    }
    
    public Vector minus(Vector other) {
        if(length != other.length) throw new IllegalArgumentException("Dimension mismatch: " + length + " vs " + other.length);
        
        double[] resultValues = new double[length];
        for(int i=0;i<length;i++) {
            resultValues[i] = values[i] - other.values[i];
        }
        
        return new Vector(resultValues);
    }
    
    public Vector scale(double lambda) {
        
        double[] resultValues = new double[length];
        for(int i=0;i<length;i++) {
            resultValues[i] = lambda * values[i];
        }
        
        return new Vector(resultValues);
    }
    
    public Matrix multiply(Vector other) {
        
        double[][] resultValues = new double[length][other.length];
        for(int rowIndex=0;rowIndex<length;rowIndex++) {
            for(int columnIndex=0;columnIndex<other.length;columnIndex++) {
                resultValues[rowIndex][columnIndex] = get(rowIndex) * other.get(columnIndex);
            }
        }
        return new Matrix(resultValues);
    }
    
    public double scalarProduct(Vector other) {
        if(length != other.length) throw new IllegalArgumentException("Dimension mismatch: " + length + " vs " + other.length);
        double sum = 0;
        for(int i=0;i<length;i++) {
            sum += get(i) * other.get(i);
        }
        return sum;
    }
    
    public double pseudoScalarProduct(Vector other) {
        double sum = 0;
        for(int i=0;i<Math.min(length, other.length);i++) {
            sum += get(i) * other.get(i);
        }
        return sum;
    }
    
    public Matrix asMatrix() {
        double[][] values = new double[length][1];
        for(int i=0;i<length;i++) {
            values[i][0] = this.values[i];
        }
        return new Matrix(values);
    }
    
    public Vector map(Function<Integer, Double> mapper) {
        double[] resultValues = new double[length];
        for(int i=0;i<length;i++) {
            resultValues[i] = mapper.apply(i);
        }
        return new Vector(resultValues);
    }
    
    @Override
    public String toString() {
        return toString(2);
    }
    public String toString(int fractionDigits) {
        String formatPattern = "%." + fractionDigits + "f";
        return DoubleStream.of(values).mapToObj(v -> String.format(formatPattern, v)).collect(joining(" ", "[", "]"));
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + length;
        result = prime * result + Arrays.hashCode(values);
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null) return false;
        if (getClass() != obj.getClass()) return false;
        Vector other = (Vector) obj;
        return IntStream.range(0, length).allMatch(index -> Math.abs(get(index) - other.get(index)) < EPSILON);
    }

}
