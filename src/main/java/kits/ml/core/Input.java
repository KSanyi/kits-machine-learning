package kits.ml.core;

public record Input(double[] values) {

    public Input(double ... values) {
        this.values = values;
    }

    public int dimension() {
        return values.length;
    }
    
    public double get(int i) {
        return values[i];
    }

}
