import javax.swing.*;
import java.awt.*;
import java.awt.event.InputMethodEvent;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.Base64;
import java.util.List;
import java.util.ArrayList;
import java.util.function.UnaryOperator;

public class FormDots extends JFrame implements Runnable, MouseListener {

    static class Point {

        public int x;
        public int y;
        public int type;

        public Point(int x, int y, int type) {
            this.x = x;
            this.y = y;
            this.type = type;
        }

    }

    private static final int devider = 4;


    private final int w = 1280;
    private final int h = 720;

    private BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
    private BufferedImage pimg = new BufferedImage(w / devider, h / devider, BufferedImage.TYPE_INT_RGB);

    private NeuralNetwork nn;
    private TeacherBackPropagation teacher;

    private List<Point> points = new ArrayList<>();

    public FormDots() {
        UnaryOperator<Double> sigmoid = (UnaryOperator<Double> & Serializable) x ->  1 / (1 + Math.exp(-x));
        UnaryOperator<Double> dsigmoid = (UnaryOperator<Double> & Serializable) y -> y * (1 - y);
        nn = new NeuralNetwork(sigmoid, 2, 32, 32, 2);
        teacher = new TeacherBackPropagation(nn, dsigmoid);

        this.setSize(w + 16, h + 38);
        this.setVisible(true);
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.setLocation(50, 50);
        this.add(new JLabel(new ImageIcon(img)));
        addMouseListener(this);
    }

    @Override
    public void run() {
        while (true) {
            this.repaint();
        }
    }
    double[][][] deltaws = null;
    @Override
    public void paint(Graphics g) {
        if(points.size() > 0) {
            double errorSum = 0;
            int cycles = 300;
            for (int k = 0; k < cycles; k++) {
                Point p = points.get(k % points.size()); // Проходимся по каждой точке
                double nx = (double) p.x / w - 0.5; // Получение значение координаты точки от -1 до 1
                double ny = (double) p.y / h - 0.5; // Получение значение координаты точки от -1 до 1
                double[] output1 = nn.feedForward(new double[]{nx, ny});

                // Указываем ожидаемые значения конечных слоёв
                double[] targets = new double[2];
                if (p.type == 0) targets[0] = 1;
                else targets[1] = 1;
                for(int kk=0;kk<2;kk++){
                    errorSum += (targets[kk] - output1[kk]) * (targets[kk] - output1[kk]);
                }
                deltaws = teacher.backpropagation(targets, 0.01, -0.2, deltaws);
            }
            System.out.println("Error: " + errorSum / cycles);
        }
        for (int i = 0; i < w / devider; i++) {
            for (int j = 0; j < h / devider; j++) {
                double nx = (double) i / w * devider - 0.5;
                double ny = (double) j / h * devider - 0.5;
                double[] outputs = nn.feedForward(new double[]{nx, ny});
                double green = Math.max(0, Math.min(1, outputs[0] - outputs[1] + 0.5));
                double blue = 1 - green;
                green = 0.3 + green * 0.5;
                blue = 0.5 + blue * 0.5;
                int color = (100 << 16) | ((int)(green * 255) << 8) | (int)(blue * 255);
                pimg.setRGB(i, j, color);
            }
        }
        Graphics ig = img.getGraphics();
        ig.drawImage(pimg, 0, 0, w, h, this);
        for (Point p : points) {
            ig.setColor(Color.WHITE);
            ig.fillOval(p.x - 3, p.y - 3, 26, 26);
            if (p.type == 0) ig.setColor(Color.GREEN);
            else ig.setColor(Color.BLUE);
            ig.fillOval(p.x, p.y, 20, 20);
        }
        g.drawImage(img, 8, 30, w, h, this);
    }

    @Override
    public void mouseClicked(MouseEvent e) {

    }

    @Override
    public void mousePressed(MouseEvent e){
        int type = 0;
        if(e.getButton() == 3) type = 1;
        points.add(new Point(e.getX() - 16, e.getY() - 38, type));
    }

    @Override
    protected void processInputMethodEvent(InputMethodEvent e) {
        System.out.println(e.getText().toString());
        super.processInputMethodEvent(e);
    }

    @Override
    public boolean keyDown(Event evt, int key) {
        System.out.println(key);
        return super.keyDown(evt, key);
    }

    @Override
    public void mouseReleased(MouseEvent e) {

    }

    @Override
    public void mouseEntered(MouseEvent e) {

    }

    @Override
    public void mouseExited(MouseEvent e) {

    }
}