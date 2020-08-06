package kits.ml.core;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class MultiClassificationModel {

    private final List<LogisticRegressionModel> models = new ArrayList<>();

    private final int nrOfClasses;

    public MultiClassificationModel(int inputDimension, int nrOfClasses) {
        this(inputDimension, nrOfClasses, 0);
    }

    public MultiClassificationModel(int inputDimension, int nrOfClasses, double lambda) {
        this.nrOfClasses = nrOfClasses;

        for (int i = 0; i < nrOfClasses; i++) {
            models.add(new LogisticRegressionModel(inputDimension, 1000, lambda));
        }
    }

    public void setModelParams(double[][] params) {
        for (int i = 0; i < nrOfClasses; i++) {
            models.get(i).setParameters(params[i]);
        }
    }

    public void learn(List<LearningData> learningDataSet) {
        for (int classIndex = 1; classIndex <= nrOfClasses; classIndex++) {
            models.get(classIndex - 1).learn(transformLearningDataSetForClass(learningDataSet, classIndex));
        }
    }

    private static List<LearningData> transformLearningDataSetForClass(List<LearningData> learningDataSet, int classIndex) {
        return learningDataSet.stream().map(learningData -> transformLearningDataForClass(learningData, classIndex)).collect(Collectors.toList());
    }

    private static LearningData transformLearningDataForClass(LearningData learningData, int classIndex) {
        int newOutput = learningData.output == classIndex ? 1 : 0;
        return new LearningData(learningData.input, newOutput);
    }

    public int predict(Input input) {
        double maxValue = 0;
        int indexForMaxValue = -1;
        for (int classIndex = 1; classIndex <= nrOfClasses; classIndex++) {
            double output = models.get(classIndex - 1).calculateOutput(input);
            // System.out.println("Probability of class " + classIndex + ": " + output);
            if (output > maxValue) {
                maxValue = output;
                indexForMaxValue = classIndex;
            }
        }

        return indexForMaxValue;
    }

}
