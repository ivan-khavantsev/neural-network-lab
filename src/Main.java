import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.lang.reflect.Field;
import java.util.function.UnaryOperator;

public class Main {

    public static void main(String[] args) throws Throwable {
        //new Thread(new FormDots()).start();
        digits();
    }

    private static void digits() throws Throwable {
        NeuralNetwork nn;

        UnaryOperator<Double> sigmoid = (UnaryOperator<Double> & Serializable) x -> 1 / (1 + Math.exp(-x));
        UnaryOperator<Double> dsigmoid = (UnaryOperator<Double> & Serializable) y -> y * (1 - y);

        boolean continueTrain = true;
        boolean save = true;

        nn = (NeuralNetwork) deserializeObjectFromFile("nn-digits.data");

        if (nn == null) {
            nn = new NeuralNetwork(sigmoid, 784, 32, 32, 10);
        }

        TeacherBackPropagation teacher = new TeacherBackPropagation(nn, dsigmoid);

        if (continueTrain) {
            int samples = 60000;
            int[] digits;
            double[][] inputs;

            digits = (int[]) deserializeObjectFromFile("TD_digits.data");
            inputs = (double[][]) deserializeObjectFromFile("TD_inputs.data");

            if(inputs == null){
                inputs = new double[samples][784];
            }

            if(digits == null){
                digits = new int[samples];
                BufferedImage[] images = new BufferedImage[samples];
                File[] imagesFiles = new File("./train").listFiles();
                for (int i = 0; i < samples; i++) {
                    images[i] = ImageIO.read(imagesFiles[i]);
                    digits[i] = Integer.parseInt(imagesFiles[i].getName().charAt(0) + "");
                }
                for (int i = 0; i < samples; i++) {
                    for (int x = 0; x < 28; x++) {
                        for (int y = 0; y < 28; y++) {
                            inputs[i][x + y * 28] = (images[i].getRGB(x, y) & 0xff) / 255.0;
                        }
                    }
                }
                serializeToFile(digits, "TD_digits.data");
                serializeToFile(inputs, "TD_inputs.data");
            }

            int epochs = 10000;
            double learningRate = 0.0012;
            for (int i = 1; i < epochs; i++) {
                int right = 0;
                double errorSum = 0;
                int batchSize = 100;
                for (int j = 0; j < batchSize; j++) {
                    int imgIndex = (int) (Math.random() * samples);
                    double[] targets = new double[10];
                    int digit = digits[imgIndex];
                    targets[digit] = 1;

                    double[] outputs = nn.feedForward(inputs[imgIndex]);
                    int maxDigit = 0;
                    double maxDigitWeight = -1;
                    for (int k = 0; k < 10; k++) {
                        if (outputs[k] > maxDigitWeight) {
                            maxDigitWeight = outputs[k];
                            maxDigit = k;
                        }
                    }
                    if (digit == maxDigit) right++;
                    for (int k = 0; k < 10; k++) {
                        errorSum += (targets[k] - outputs[k]) * (targets[k] - outputs[k]);
                    }
                    teacher.backpropagation(targets, learningRate);
                }
                System.out.println("epoch: " + i + ". correct: " + right + ". error: " + errorSum);
            }
        }

        if(save) {
           serializeToFile(nn, "nn-digits.data");
        }
        FormDigits f = new FormDigits(nn);
        new Thread(f).start();
    }


    public static byte[] serialize(Object object) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        oos.writeObject(object);
        oos.close();
        return baos.toByteArray();
    }

    public static void serializeToFile(Object object, String filename) throws IOException {
        byte[] bytes = serialize(object);
        try(FileOutputStream fileOutputStream = new FileOutputStream(filename)){
            fileOutputStream.write(bytes);
        }
    }

    public static Object deserializeObject(byte[] bytes) throws IOException, ClassNotFoundException {
        try(ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(bytes))){
            return ois.readObject();
        }
    }

    public static Object deserializeObjectFromFile(String filename) throws IOException, ClassNotFoundException{
        File file = new File(filename);
        if(file.exists()){
            try(FileInputStream fis = new FileInputStream(file)){
                return deserializeObject(fis.readAllBytes());
            }
        }
        return null;
    }


    public static NeuralNetwork deserialize(byte[] bytes) throws IOException, ClassNotFoundException {
        return (NeuralNetwork) deserializeObject(bytes);
    }

}