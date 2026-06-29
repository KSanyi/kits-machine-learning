package kits.ml.neuralnet;

import static kits.ml.neuralnet.ActivationFunction.StandardActivationFunction.RELU;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class ActivationFunctionTest {

    private static final double DELTA = 1e-9;

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
}
