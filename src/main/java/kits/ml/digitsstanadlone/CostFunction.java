package kits.ml.digitsstanadlone;

public interface CostFunction {

    double calculateCost(Vector output, Vector expectedOutput);
    
}

class QuadraticCostFunction implements CostFunction {

    @Override
    public double calculateCost(Vector output, Vector expectedOutput) {
        return output.minus(expectedOutput).normSquared();
    }
    
}

