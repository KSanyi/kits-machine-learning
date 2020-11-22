package kits.ml.core.math.linalg;

import java.util.ArrayList;
import java.util.List;

public class GramSchmidtCalculator {

    private static final double EPSILON = 0.00001;
    
    public static Matrix createOrtoNormalBaseMatrix(Matrix A) {
        
        if(A.getNrColumns() > A.getNrRows()) throw new IllegalArgumentException("A.getNrColumns() > A.getNrRows()");

        List<Vector> resultColumns = new ArrayList<>();
        
        for(int i=0;i<A.getNrColumns();i++) {
            Vector column = A.getColumnVector(i);
            resultColumns.add(ortoNormalize(column, resultColumns));
        }
        
        return Matrix.fromColumnVectors(resultColumns);
    }

    private static Vector ortoNormalize(Vector vector, List<Vector> ortoNormalVectors) {
        Vector result = vector;
        for(Vector ortonormalVector : ortoNormalVectors) {
            result = result.minus(ortonormalVector.multiply(ortonormalVector.scalarProduct(result)));
        }
        
        double norm = result.norm();
        if(Math.abs(norm) > EPSILON) {
            return result.multiply(1 / result.norm());
        } else {
            throw new IllegalArgumentException("Columns of A are not linearly independent");            
        }
    }
    
}
