import java.io.*;
import java.util.function.UnaryOperator;

public class NeuralNetwork implements Serializable {

    public Layer[] layers;
    public UnaryOperator<Double> activation;

    public NeuralNetwork(UnaryOperator<Double> activation, int... sizes) {

        this.activation = activation;

        layers = new Layer[sizes.length];
        for (int i = 0; i < sizes.length; i++) {
            int nextSize = 0;
            if (i < sizes.length - 1) nextSize = sizes[i + 1];
            layers[i] = new Layer(sizes[i], nextSize);
            for (int j = 0; j < sizes[i]; j++) {
                layers[i].biases[j] = Math.random() * 2.0 - 1.0; // -1.0 to 1.0
                for (int k = 0; k < nextSize; k++) {
                    layers[i].weights[j][k] = Math.random() * 2.0 - 1.0; // -1.0 to 1.0
                }
            }
        }
    }

    public double[] feedForward(double[] inputs) {
        System.arraycopy(inputs, 0, layers[0].neurons, 0, inputs.length);
        for (int l = 1; l < layers.length; l++) {
            Layer cl = layers[l - 1];
            Layer nl = layers[l];
            for (int i = 0; i < nl.size; i++) {
                nl.neurons[i] = 0;
                for (int j = 0; j < cl.size; j++) {
                    nl.neurons[i] += cl.neurons[j] * cl.weights[j][i];
                }
                nl.neurons[i] += nl.biases[i];
                nl.neurons[i] = activation.apply(nl.neurons[i]);
            }
        }
        return layers[layers.length - 1].neurons;
    }
}
