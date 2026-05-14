package kits.ml.core.math.optimization;

import java.util.function.Function;

import kits.ml.core.math.linalg.Vector;

public interface GradientOptimizer {

    Vector optimize(Vector startingVector, Function<Vector, Double> function, Function<Vector, Vector> gradient);
    
}
