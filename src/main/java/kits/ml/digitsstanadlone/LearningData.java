package kits.ml.digitsstanadlone;

import java.util.List;

record LearningData(Vector input, double output) {}

record DataSets(List<LearningData> trainingData, List<LearningData> testData) {}
