package kits.ml.application.digitdetection;

import static java.lang.Integer.MAX_VALUE;

import java.io.File;
import java.io.IOException;
import java.util.List;

import kits.ml.core.DataPoint;
import kits.ml.core.KNearestNeighbour;
import kits.ml.core.math.linalg.Vector;

public class KNearestNeighbourDigitDetector extends BaseDigitDetector {
    
    private final int k;
    
    public KNearestNeighbourDigitDetector(int k) {
        this.k = k;
    }
    
    private List<DataPoint> dataPoints = List.of();
    
    public void learn(File trainingSet) throws IOException {
        dataPoints = loadDataPoints(trainingSet, MAX_VALUE);
    }

    public int detect(File digitFile) throws IOException {
        Vector output = KNearestNeighbour.find(createInputVector(digitFile), dataPoints, k);
        
        return convertoToDigit(output);
    }
    
}
