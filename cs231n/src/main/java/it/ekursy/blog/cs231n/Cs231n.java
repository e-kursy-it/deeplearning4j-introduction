package it.ekursy.blog.cs231n;

import static org.nd4j.linalg.ops.transforms.Transforms.*;

import java.awt.*;
import java.util.List;
import javax.swing.*;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.math.plot.Plot2DPanel;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Cs231n {

    private static final Logger LOG = LogManager.getLogger();

    private int N = 100; // number of points per class
    private int D = 2; // dimensionality
    private int K = 3; // number of classes

    private final List<Color> colors = List.of(Color.RED, Color.GREEN, Color.BLUE);

    public INDArray generateData() {
        INDArray X = Nd4j.zeros(K, D, N);// # data matrix (each row = single example)
        INDArray y = Nd4j.zeros(K, N);

        for (int j = 0; j < K; j++) {
            // r = np.linspace(0.0,1,N) # radius
            INDArray r = Nd4j.linspace(0.0, 1, N);// # radius

            // t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
            INDArray t = Nd4j
                    .linspace(j * 4, (j + 1) * 4, N)
                    .add(Nd4j.randn(new int[]{N})
                            .mul(0.2));// # theta

            // X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]

            X.getRow(j).putRow(0, r.mul(sin(t)));
            X.getRow(j).putRow(1, r.mul(cos(t)));

            y.putRow(j, Nd4j.zeros(N).addi(j));
        }

        y = y.reshape(N * K);

        return X;
    }

    public void visualizeData(INDArray X) {
        // create your PlotPanel (you can use it as a JPanel)
        Plot2DPanel plot = new Plot2DPanel();

        for (int i = 0; i < K; i++) {
            double[] x = X.getRow(i).getRow(0).toDoubleVector();
            double[] y = X.getRow(i).getRow(1).toDoubleVector();

            // add a line plot to the PlotPanel
            plot.addScatterPlot("my plot", colors.get(i), x, y);
        }

        // put the PlotPanel in a JFrame, as a JPanel
        JFrame frame = new JFrame("a plot panel");
        frame.setContentPane(plot);
        frame.setVisible(true);
        frame.setSize(400, 400);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    }

    public void trainingSoftMaxClassifier() {
        INDArray W = Nd4j.randn(D, K).muli(0.01);
        INDArray b = Nd4j.zeros(1, K);

        INDArray X = generateData().reshape(N * K, D);

        // scores = np.dot(X, W) + b
        // bo contains only zeros so I skipped it...
        INDArray scores = X.mmul(W).addRowVector(b);

        long num_examples = X.shape()[0];
        INDArray exp_scores = Transforms.exp(scores);

        // removing second param to sum
        // https://github.com/e-kursy-it/JavaRNN/blob/3863d13b1b06b63cc85bcd594622c350cd0e36f8/src/main/java/com/guilherme/charRNN/CharRNN.java#L275
        INDArray probs = exp_scores.div(Nd4j.sum(exp_scores));

        LOG.info("Probs shape: {}", probs.shape());

        // correct_logprobs = -np.log(probs[range(num_examples),y])
//        correct_logprobs = -Transforms.log(probs.)
    }

    public static void main(String[] args) {
        Cs231n cs231N = new Cs231n();
        //INDArray data = cs231N.generateData();
        cs231N.trainingSoftMaxClassifier();
    }
}