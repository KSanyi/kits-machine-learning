package kits.ml.neuralnet;

import static kits.ml.neuralnet.ActivationFunction.StandardActivationFunction.RELU;
import static kits.ml.neuralnet.ActivationFunction.StandardActivationFunction.SOFTMAX;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import kits.ml.core.math.linalg.Vector;
import org.junit.jupiter.api.Test;

public class ActivationFunctionTest {

    private static final double DELTA = 1e-6;

    @Test
    public void relu_positive_input() {
        assertEquals(3.5, RELU.apply(3.5), DELTA);
    }

    @Test
    public void relu_zero_input() {
        assertEquals(0.0, RELU.apply(0.0), DELTA);
    }

    @Test
    public void relu_negative_input() {
        assertEquals(0.0, RELU.apply(-2.0), DELTA);
    }

    @Test
    public void relu_derivative_positive_activated_value() {
        assertEquals(1.0, RELU.derivative(5.0), DELTA);
    }

    @Test
    public void relu_derivative_zero_activated_value() {
        // activated value = 0 means pre-activation was <= 0
        assertEquals(0.0, RELU.derivative(0.0), DELTA);
    }

    @Test
    public void relu_derivative_negative_activated_value_is_impossible_but_returns_zero() {
        assertEquals(0.0, RELU.derivative(-1.0), DELTA);
    }

    // --- Softmax ---

    @Test
    public void softmax_outputs_sum_to_one() {
        Vector result = SOFTMAX.applyVector(new Vector(1.0, 2.0, 3.0));
        double sum = result.stream().sum();
        assertEquals(1.0, sum, DELTA);
    }

    @Test
    public void softmax_preserves_order() {
        Vector result = SOFTMAX.applyVector(new Vector(1.0, 2.0, 3.0));
        // higher input -> higher output probability
        assert result.get(0) < result.get(1);
        assert result.get(1) < result.get(2);
    }

    @Test
    public void softmax_uniform_inputs_give_equal_outputs() {
        Vector result = SOFTMAX.applyVector(new Vector(1.0, 1.0, 1.0));
        assertEquals(1.0 / 3, result.get(0), DELTA);
        assertEquals(1.0 / 3, result.get(1), DELTA);
        assertEquals(1.0 / 3, result.get(2), DELTA);
    }

    @Test
    public void softmax_numerically_stable_with_large_inputs() {
        // would overflow without the max-subtraction trick
        Vector result = SOFTMAX.applyVector(new Vector(1000.0, 1001.0, 1002.0));
        double sum = result.stream().sum();
        assertEquals(1.0, sum, DELTA);
    }

    @Test
    public void softmax_apply_scalar_throws() {
        assertThrows(UnsupportedOperationException.class, () -> SOFTMAX.apply(1.0));
    }

    @Test
    public void softmax_derivative_scalar_throws() {
        assertThrows(UnsupportedOperationException.class, () -> SOFTMAX.derivative(0.5));
    }

    @Test
    public void softmax_output_delta_equals_prediction_minus_target() {
        Vector predicted = SOFTMAX.applyVector(new Vector(2.0, 1.0, 0.1));
        Vector target = new Vector(1.0, 0.0, 0.0);
        Vector delta = SOFTMAX.outputDelta(predicted, target, CostFunction.StandardCostFunction.CROSS_ENTROPY);
        for (int i = 0; i < 3; i++) {
            assertEquals(predicted.get(i) - target.get(i), delta.get(i), DELTA);
        }
    }
}
