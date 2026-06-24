package kits.ml.regression;

import java.util.List;

import kits.ml.core.LearningData;
import kits.ml.core.math.linalg.Vector;

public interface MLModel {

    void learn(List<LearningData> learningDataSet);

    double calculateOutput(Vector input);

    double calculateCost(List<LearningData> learningDataSet);

}
