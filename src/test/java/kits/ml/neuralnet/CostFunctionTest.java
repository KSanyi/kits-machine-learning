package kits.ml.neuralnet;

import static kits.ml.neuralnet.CostFunction.StandardCostFunction.CATEGORICAL_CROSS_ENTROPY;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import kits.ml.core.math.linalg.Vector;
import org.junit.jupiter.api.Test;

public class CostFunctionTest {

    private static final double DELTA = 1e-9;

    @Test
    public void categorical_cross_entropy_perfect_prediction() {
        // predicted assigns probability 1.0 to the true class -> cost = -log(1) = 0
        Vector predicted = new Vector(0.0, 1.0, 0.0);
        Vector trueLabel = new Vector(0.0, 1.0, 0.0);
        assertEquals(0.0, CATEGORICAL_CROSS_ENTROPY.cost(predicted, trueLabel), DELTA);
    }

    @Test
    public void categorical_cross_entropy_known_value() {
        // true class is index 0, predicted probability = 0.9 -> cost = -log(0.9)
        Vector predicted = new Vector(0.9, 0.05, 0.05);
        Vector trueLabel = new Vector(1.0, 0.0, 0.0);
        assertEquals(-Math.log(0.9), CATEGORICAL_CROSS_ENTROPY.cost(predicted, trueLabel), DELTA);
    }

    @Test
    public void categorical_cross_entropy_low_confidence_costs_more() {
        Vector predicted1 = new Vector(0.9, 0.05, 0.05);
        Vector predicted2 = new Vector(0.5, 0.25, 0.25);
        Vector trueLabel  = new Vector(1.0, 0.0,  0.0);
        double cost1 = CATEGORICAL_CROSS_ENTROPY.cost(predicted1, trueLabel);
        double cost2 = CATEGORICAL_CROSS_ENTROPY.cost(predicted2, trueLabel);
        assert cost1 < cost2 : "higher confidence should have lower cost";
    }

    @Test
    public void categorical_cross_entropy_clamps_zero_probability() {
        // zero probability would be log(0) = -Inf; clamped to 1e-15 so cost is finite
        Vector predicted = new Vector(0.0, 1.0, 0.0);
        Vector trueLabel = new Vector(1.0, 0.0, 0.0);
        double cost = CATEGORICAL_CROSS_ENTROPY.cost(predicted, trueLabel);
        assert Double.isFinite(cost) : "cost must be finite even for zero probability";
    }

    @Test
    public void categorical_cross_entropy_gradient_throws() {
        assertThrows(UnsupportedOperationException.class, () ->
            CATEGORICAL_CROSS_ENTROPY.gradient(new Vector(0.7, 0.2, 0.1), new Vector(1.0, 0.0, 0.0)));
    }
}
