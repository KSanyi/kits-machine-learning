package kits.ml.core;

import java.util.List;

import kits.ml.core.math.linalg.Vector;

public interface MLModel {

    void learn(List<LearningData> learningDataSet);

    double calculateOutput(Vector input);

    double calculateCost(List<LearningData> learningDataSet);

}
