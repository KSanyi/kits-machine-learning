package kits.ml.core.math.linalg;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;

public class GramSchmidtCalculator {

    private static final double EPSILON = 0.00001;
    
    public static Matrix createOrtoNormalBaseMatrix(Matrix A) {
        
        if(A.getNrColumns() > A.getNrRows()) throw new IllegalArgumentException("A.getNrColumns() > A.getNrRows()");

        List<Vector> resultColumns = new ArrayList<>();
        
        for(int i=0;i<A.getNrColumns();i++) {
            Vector column = A.getColumnVector(i);
            resultColumns.add(ortoNormalize(column, resultColumns));
        }
        
        return MatrixFactory.fromColumnVectors(resultColumns);
    }

    private static Vector ortoNormalize(Vector vector, List<Vector> ortoNormalVectors) {
        Vector result = vector;
        for(Vector ortonormalVector : ortoNormalVectors) {
            result = result.minus(ortonormalVector.scale(ortonormalVector.scalarProduct(result)));
        }
        
        double norm = result.norm();
        if(Math.abs(norm) > EPSILON) {
            return result.scale(1 / result.norm());
        } else {
            throw new IllegalArgumentException("Columns of A are not linearly independent");            
        }
    }
    
    public static Decomposition.QR createQRDecomposition(Matrix A) {
        
        if(A.getNrColumns() > A.getNrRows()) throw new IllegalArgumentException("A.getNrColumns() > A.getNrRows()");

        List<Vector> qColumns = new ArrayList<>();
        List<Vector> rColumns = new ArrayList<>();
        
        for(int i=0;i<A.getNrColumns();i++) {
            Vector column = A.getColumnVector(i);
            var qColumnAndRColumn = ortoNormalizeAndCreateRColumn(column, qColumns);
            qColumns.add(qColumnAndRColumn.getLeft());
            rColumns.add(qColumnAndRColumn.getRight());
        }
        
        return new Decomposition.QR(MatrixFactory.fromColumnVectors(qColumns), MatrixFactory.fromColumnVectors(rColumns));
    }
    
    private static Pair<Vector, Vector> ortoNormalizeAndCreateRColumn(Vector vector, List<Vector> ortoNormalVectors) {
        Vector qColumn = vector;
        Vector rColumn = new Vector(vector.length());
        for(int i=0;i<ortoNormalVectors.size();i++) {
            Vector ortoNormalVector = ortoNormalVectors.get(i);
            double prod = ortoNormalVector.scalarProduct(qColumn);
            qColumn = qColumn.minus(ortoNormalVector.scale(prod));
            rColumn.set(i, prod);
        }
        
        double norm = qColumn.norm();
        if(Math.abs(norm) > EPSILON) {
            qColumn = qColumn.scale(1 / qColumn.norm());
            rColumn.set(ortoNormalVectors.size(), vector.scalarProduct(qColumn));
            return Pair.of(qColumn, rColumn);
        } else {
            throw new IllegalArgumentException("Columns of A are not linearly independent");            
        }
    }
    
}
