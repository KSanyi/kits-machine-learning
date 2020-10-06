package kits.ml.core.math.linalg;

public record LUDecomposition(Matrix L, Matrix U, Matrix P) {

    public LUDecomposition(Matrix L, Matrix U) {
        this(L, U, null);
    }
    
}
