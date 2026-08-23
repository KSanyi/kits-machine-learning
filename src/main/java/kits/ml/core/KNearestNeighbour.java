package kits.ml.core;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import kits.ml.core.math.linalg.Vector;
import kits.ml.util.FrequencyMap;

public class KNearestNeighbour {

    public static Vector find(Vector inputVector, List<DataPoint> dataPoints, int k) {
        
        KNearestHolder kNearestHolder = new KNearestHolder(k);
        for(DataPoint dataPoint : dataPoints) {
            kNearestHolder.addIfNear(dataPoint, distance(inputVector, dataPoint.input()));
        }
        
        return kNearestHolder.vote();
    }
    
    private static double distance(Vector x, Vector y) {
        return x.minus(y).normSquared();
    }
    
    private static class KNearestHolder {
        final int k;        
        final Map<DataPoint, Double> dataPointToDistance = new HashMap<>();
        
        DataPoint farthestDatapoint;
        double farthestDatapointDistance;
        
        KNearestHolder(int k) {
            this.k = k;
        }

        void addIfNear(DataPoint dataPoint, double distance) {
            if(dataPointToDistance.size() < k) {
                dataPointToDistance.put(dataPoint, distance);
                refreshFarthest();
            } else {
                if(distance < farthestDatapointDistance) {
                    dataPointToDistance.remove(farthestDatapoint);
                    dataPointToDistance.put(dataPoint, distance);
                    refreshFarthest();
                }
            }
        }
        
        private void refreshFarthest() {
            farthestDatapoint = null;
            farthestDatapointDistance = 0;
            for(var entry : dataPointToDistance.entrySet()) {
                if(entry.getValue() > farthestDatapointDistance) {
                    farthestDatapoint = entry.getKey();
                    farthestDatapointDistance = entry.getValue();
                }
            }
        }

        Vector vote() {
            FrequencyMap<Vector> frequencyMap = new FrequencyMap<Vector>();
            for(DataPoint dataPoint : dataPointToDistance.keySet()) {
                frequencyMap.put(dataPoint.output());
            }
            return frequencyMap.mostFrequent();
        }
        
    }

}
