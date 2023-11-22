import java.util.function.UnaryOperator;

public class TeacherBackPropagation {

    public UnaryOperator<Double> derivative;
    public NeuralNetwork nn;
    
    public TeacherBackPropagation(NeuralNetwork nn, UnaryOperator<Double> derivative){
        this.nn = nn;
        this.derivative = derivative;
    }


    public double[][][] backpropagation(double[] targets, double learningRate, double moment, double[][][] deltaWs) {
        if(deltaWs == null){
            deltaWs = new double[nn.layers.length][][];
        }

        Layer ol = nn.layers[nn.layers.length - 1]; // Выходной слой

        double[] errors = new double[ol.size];
        for (int i = 0; i < ol.size; i++) {
            // Сначала ищем ошибки последнего-выходного слоя
            errors[i] = targets[i] - nn.layers[nn.layers.length - 1].neurons[i];
        }

        // Повторяем всё для каждого скрытого слоя
        double[] gradients;
        for (int k = nn.layers.length - 2; k >= 0; k--) {
            Layer nl = nn.layers[k]; // New layer, следующий слой (в обратном порядке идём)
            Layer cl = nn.layers[k + 1]; // Current layer, текущий слой

            gradients = new double[cl.size];
            for (int i = 0; i < cl.size; i++) {
                gradients[i] = errors[i] * derivative.apply(cl.neurons[i]);
            }

            double[] errorsNext = new double[nl.size];
            for (int i = 0; i < nl.size; i++) {
                for (int j = 0; j < cl.size; j++) {
                    errorsNext[i] += nl.weights[i][j] * errors[j];
                }
            }

            errors = new double[nl.size];
            System.arraycopy(errorsNext, 0, errors, 0, nl.size);

            // Вычисляем и устанавливаем новые веса для следующего слоя
            for (int i = 0; i < cl.size; i++) {
                for (int j = 0; j < nl.size; j++) {
                    if(deltaWs[k] == null){
                        deltaWs[k] = new double[cl.size][nl.size];
                    }

                    double deltaW = gradients[i] * (nl.neurons[j] * learningRate) - (moment * deltaWs[k][i][j]);
                    deltaWs[k][i][j] = deltaW;
                    nl.weights[j][i] = nl.weights[j][i] + deltaW;
                }
            }

            // Обновляем байесы
            for (int i = 0; i < cl.size; i++) {
                cl.biases[i] += gradients[i];
            }
        }
        return deltaWs;
    }
}
