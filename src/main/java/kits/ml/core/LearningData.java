package kits.ml.core;

import kits.ml.core.math.linalg.Vector;

public record LearningData(Vector input, double output) {

    @Override
    public String toString() {
        return input + " -> " + output;
    }

}
