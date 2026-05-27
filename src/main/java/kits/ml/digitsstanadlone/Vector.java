package kits.ml.digitsstanadlone;

import static java.util.stream.Collectors.joining;

import java.util.function.Function;
import java.util.stream.DoubleStream;

public class Vector {

    private final int length;
    
    private final double[] values;

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
    
    public double normSquared() {
        return this.scalarProduct(this);
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
    
    public void plusThis(Vector other) {
        if(length != other.length) throw new IllegalArgumentException("Dimension mismatch: " + length + " vs " + other.length);
        
        for(int i=0;i<length;i++) {
            values[i] = values[i] + other.values[i];
        }
    }
    
    public void minusThis(Vector other) {
        if(length != other.length) throw new IllegalArgumentException("Dimension mismatch: " + length + " vs " + other.length);
        
        for(int i=0;i<length;i++) {
            values[i] = values[i] - other.values[i];
        }
    }
    
    public Vector scale(double lambda) {
        
        double[] resultValues = new double[length];
        for(int i=0;i<length;i++) {
            resultValues[i] = lambda * values[i];
        }
        
        return new Vector(resultValues);
    }
    
    public void scaleThis(double lambda) {
        
        for(int i=0;i<length;i++) {
            values[i] = lambda * values[i];
        }
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
    
    public Vector map(Function<Integer, Double> mapper) {
        double[] resultValues = new double[length];
        for(int i=0;i<length;i++) {
            resultValues[i] = mapper.apply(i);
        }
        return new Vector(resultValues);
    }
    
    public void apply(Function<Integer, Double> mapper) {
        for(int i=0;i<length;i++) {
            values[i] = mapper.apply(i);
        }
    }
    
    @Override
    public String toString() {
        return toString(2);
    }
    
    public String toString(int fractionDigits) {
        String formatPattern = "%." + fractionDigits + "f";
        return stream().mapToObj(v -> String.format(formatPattern, v)).collect(joining(" ", "[", "]"));
    }

    public DoubleStream stream() {
        return DoubleStream.of(values);
    }
    
    public static Vector createZero(int length) {
        double[] data = new double[length];
        return new Vector(data);
    }
    
    public static Vector createOneHot(int length, int value) {
        Vector oneHot = Vector.createZero(length);
        oneHot.set(value, 1);
        return oneHot;
    }

}
