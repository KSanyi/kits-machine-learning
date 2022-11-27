package kits.ml.core;

public record LearningData(Input input, double output) {

    @Override
    public String toString() {
        return input + " -> " + output;
    }

}
