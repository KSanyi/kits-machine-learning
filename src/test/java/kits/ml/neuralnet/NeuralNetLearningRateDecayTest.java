package kits.ml.neuralnet;

import static kits.ml.neuralnet.ActivationFunction.StandardActivationFunction.SIGMOID;
import static kits.ml.neuralnet.CostFunction.StandardCostFunction.CROSS_ENTROPY;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class NeuralNetLearningRateDecayTest {

    private static final double DELTA = 1e-10;

    private NeuralNet net;

    @BeforeEach
    public void setUp() {
        net = new NeuralNet(CROSS_ENTROPY, SIGMOID, 4, 2);
        net.setLearningRate(0.01);
    }

    @Test
    public void no_decay_returns_base_rate_at_every_epoch() {
        assertEquals(0.01, net.learningRateAt(0),   DELTA);
        assertEquals(0.01, net.learningRateAt(50),  DELTA);
        assertEquals(0.01, net.learningRateAt(99),  DELTA);
    }

    @Test
    public void step_decay_unchanged_before_first_step() {
        net.setLearningRateDecay(33, 0.5);
        assertEquals(0.01, net.learningRateAt(0),  DELTA);
        assertEquals(0.01, net.learningRateAt(32), DELTA);
    }

    @Test
    public void step_decay_halves_at_first_step() {
        net.setLearningRateDecay(33, 0.5);
        assertEquals(0.005, net.learningRateAt(33), DELTA);
        assertEquals(0.005, net.learningRateAt(65), DELTA);
    }

    @Test
    public void step_decay_halves_again_at_second_step() {
        net.setLearningRateDecay(33, 0.5);
        assertEquals(0.0025, net.learningRateAt(66), DELTA);
        assertEquals(0.0025, net.learningRateAt(99), DELTA);
    }

    @Test
    public void custom_factor_applied_correctly() {
        net.setLearningRateDecay(10, 0.1);
        assertEquals(0.01,     net.learningRateAt(0),  DELTA);
        assertEquals(0.001,    net.learningRateAt(10), DELTA);
        assertEquals(0.0001,   net.learningRateAt(20), DELTA);
        assertEquals(0.00001,  net.learningRateAt(30), DELTA);
    }

    @Test
    public void decay_disabled_when_steps_set_to_zero() {
        net.setLearningRateDecay(0, 0.5);
        assertEquals(0.01, net.learningRateAt(999), DELTA);
    }
}
