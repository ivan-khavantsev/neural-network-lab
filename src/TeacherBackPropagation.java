import java.util.function.UnaryOperator;

public class TeacherBackPropagation {

    public UnaryOperator<Double> derivative;
    public NeuralNetwork nn;
    
    public TeacherBackPropagation(NeuralNetwork nn, UnaryOperator<Double> derivative){
        this.nn = nn;
        this.derivative = derivative;
    }
    
    public void backpropagation(double[] targets, double learningRate) {
        double[] errors = new double[nn.layers[nn.layers.length - 1].size];
        for (int i = 0; i < nn.layers[nn.layers.length - 1].size; i++) {
            errors[i] = targets[i] - nn.layers[nn.layers.length - 1].neurons[i];
        }
        for (int k = nn.layers.length - 2; k >= 0; k--) {
            Layer l = nn.layers[k];
            Layer l1 = nn.layers[k + 1];
            double[] errorsNext = new double[l.size];
            double[] gradients = new double[l1.size];
            for (int i = 0; i < l1.size; i++) {
                gradients[i] = errors[i] * derivative.apply(nn.layers[k + 1].neurons[i]);
                gradients[i] *= learningRate;
            }
            double[][] deltas = new double[l1.size][l.size];
            for (int i = 0; i < l1.size; i++) {
                for (int j = 0; j < l.size; j++) {
                    deltas[i][j] = gradients[i] * l.neurons[j];
                }
            }
            for (int i = 0; i < l.size; i++) {
                errorsNext[i] = 0;
                for (int j = 0; j < l1.size; j++) {
                    errorsNext[i] += l.weights[i][j] * errors[j];
                }
            }
            errors = new double[l.size];
            System.arraycopy(errorsNext, 0, errors, 0, l.size);
            double[][] weightsNew = new double[l.weights.length][l.weights[0].length];
            for (int i = 0; i < l1.size; i++) {
                for (int j = 0; j < l.size; j++) {
                    weightsNew[j][i] = l.weights[j][i] + deltas[i][j];
                }
            }
            l.weights = weightsNew;
            for (int i = 0; i < l1.size; i++) {
                l1.biases[i] += gradients[i];
            }
        }
    }
}
