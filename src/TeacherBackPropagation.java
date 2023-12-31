import java.util.function.UnaryOperator;

public class TeacherBackPropagation {

    public UnaryOperator<Double> derivative;
    public NeuralNetwork nn;

    public TeacherBackPropagation(NeuralNetwork nn, UnaryOperator<Double> derivative) {
        this.nn = nn;
        this.derivative = derivative;
    }

    public void backpropagation(double[] targets, double learningRate, double moment, State state) {

        if (state.deltaWeights == null || state.deltaBieses == null) {
            state.deltaWeights = new double[nn.layers.length][][];
            state.deltaBieses = new double[nn.layers.length][];
        }

        Layer ol = nn.layers[nn.layers.length - 1]; // Выходной слой

        // Сначала сюда пишем ошибки выходного слоя
        double[] errors = new double[ol.size];
        for (int i = 0; i < ol.size; i++) {
            errors[i] = targets[i] - nn.layers[nn.layers.length - 1].neurons[i];
        }

        // Повторяем всё для каждого скрытого слоя
        double[] gradients;
        for (int nli = nn.layers.length - 2; nli >= 0; nli--) { // Next Layer Index
            Layer nl = nn.layers[nli]; // Next layer, следующий слой (в обратном порядке идём)
            int cli = nli + 1; // Current Layer Index
            Layer cl = nn.layers[cli]; // Current layer, текущий слой

            gradients = new double[cl.size];
            for (int i = 0; i < cl.size; i++) {
                // Вычисляем градиент по производной от функции активации
                gradients[i] = errors[i] * derivative.apply(cl.neurons[i]);
            }

            // Когда вычислили градиенты, ищем ошибки следующего слоя для следующей итерации цикла
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
                    if (state.deltaWeights[nli] == null) {
                        state.deltaWeights[nli] = new double[cl.size][nl.size];
                    }

                    // Изменение, которое нужно произвести в весе следующего слоя
                    double deltaWeight = gradients[i] * nl.neurons[j] * learningRate + moment * state.deltaWeights[nli][i][j];
                    // Обновляем вес
                    nl.weights[j][i] = nl.weights[j][i] + deltaWeight;
                    // Сохраняем все дельты в статус, для следующего раза
                    state.deltaWeights[nli][i][j] = deltaWeight;
                }
            }

            // Обновляем байесы
            for (int i = 0; i < cl.size; i++) {
                if (state.deltaBieses[cli] == null) {
                    state.deltaBieses[cli] = new double[cl.size];
                }
                // Вычисляем дельту для байеса текущего слоя
                double deltaBias = gradients[i] * learningRate + moment * state.deltaBieses[cli][i];
                // Обновляем байес
                cl.biases[i] += deltaBias;
                // Сохраняем все дельты в статус, для следующего раза
                state.deltaBieses[cli][i] = deltaBias;
            }
        }
    }

    public static class State {
        double[][][] deltaWeights;
        double[][] deltaBieses;
    }
}
