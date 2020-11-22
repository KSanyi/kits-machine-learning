package kits.ml.core.math.linalg;

public class Decomposition {
    
    public static record LU(
            Matrix L,   // lower triangular matrix 
            Matrix U,   // upper triangular matrix
            Matrix P) { // optional permutation matrix

        public LU(Matrix L, Matrix U) {
            this(L, U, null);
        }
        
    }
    
    public static record QR(
            Matrix Q,   // ortonormal matrix
            Matrix R) { // upper triangular matrix
    }
    
}


