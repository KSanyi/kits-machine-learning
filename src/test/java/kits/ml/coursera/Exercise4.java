package kits.ml.coursera;

import java.util.Arrays;
import java.util.List;

import kits.ml.core.Input;
import kits.ml.core.LearningData;
import kits.ml.core.math.MLMath;
import kits.ml.core.math.linalg.Matrix;
import kits.ml.neuralnet.NeuralNet;

public class Exercise4 {

    static final Matrix X = new Matrix(new double[][] {
        {2.524413, -2.270407, 1.970960, -1.632063, 1.260501, -0.863710, 0.449632, -0.026554, -0.397055,  0.812717},
        {2.727892, -2.876773, 2.968075, -2.999971, 2.971822, -2.884192, 2.738836, -2.538661,  2.287675, -1.990902},
        {0.423360, -0.838246, 1.236355, -1.609719, 1.950864, -2.252962, 2.509967, -2.716735,  2.869128, -2.964095}});
        
    
    static final List<LearningData> learningDataSet = Arrays.asList(
            new LearningData(new Input( 0.1682942, -0.1922795), 2),
            new LearningData(new Input( 0.1818595, -0.1501974), 3),
            new LearningData(new Input( 0.0282240,  0.0299754), 4),
            new LearningData(new Input(-0.1513605,  0.1825891), 1),
            new LearningData(new Input(-0.1917849,  0.1673311), 2),
            new LearningData(new Input(-0.0558831, -0.0017703), 3),
            new LearningData(new Input( 0.1313973, -0.1692441), 4),
            new LearningData(new Input( 0.1978716, -0.1811157), 1),
            new LearningData(new Input( 0.0824237, -0.0264704), 2),
            new LearningData(new Input(-0.1088042,  0.1525117), 3),
            new LearningData(new Input(-0.1999980,  0.1912752), 4),
            new LearningData(new Input(-0.1073146,  0.0541812), 1),
            new LearningData(new Input( 0.0840334, -0.1327268), 2),
            new LearningData(new Input( 0.1981215, -0.1976063), 3),
            new LearningData(new Input( 0.1300576, -0.0808075), 4),
            new LearningData(new Input(-0.0575807,  0.1102853), 1));
         
    static final double[][] weights1 = new double[][] {
        { 0.84147,  0.41212, -0.96140},
        { 0.14112, -0.99999,  0.14988},
        {-0.95892,  0.42017,  0.83666},
        { 0.65699,  0.65029, -0.84622}  
    };
    
    static final double[][] weights2 = new double[][] {
        { 0.5403023, -0.9111303, -0.2751633,  0.9912028, -0.0132767},
        {-0.9899925,  0.0044257,  0.9887046, -0.2921388, -0.9036922},
        { 0.2836622,  0.9074468, -0.5477293, -0.7480575,  0.7654141},
        { 0.7539023, -0.7596879, -0.5328330,  0.9147424,  0.2666429}  
    };
    
    public static void main(String[] args) {
        task1();
        System.out.println();
        task2();
        System.out.println();
        task3();
    }
    
    private static void task1() {
        System.out.println("Task 1");
        
        NeuralNet neuralNet = new NeuralNet(Arrays.asList(weights1, weights2), 0);
        
        System.out.println(neuralNet.calculateCost(learningDataSet));
    }
    
    private static void task2() {
        System.out.println("Task 2");
        
        NeuralNet neuralNet = new NeuralNet(Arrays.asList(weights1, weights2), 1.5);
        
        System.out.println(neuralNet.calculateCost(learningDataSet));
    }
    
    private static void task3() {
        System.out.println("Task 3");
        Matrix XGrad = MLMath.sigmoidGradient(X);
        System.out.println(XGrad);
        System.out.println();
    }
}
