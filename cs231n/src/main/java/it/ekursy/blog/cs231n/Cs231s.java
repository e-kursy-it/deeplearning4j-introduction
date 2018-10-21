package it.ekursy.blog.dl4jintro;

import static org.nd4j.linalg.ops.transforms.Transforms.*;

import java.awt.*;
import java.util.List;
import javax.swing.*;

import org.math.plot.Plot2DPanel;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Cs231s {

    private int N = 100; // number of points per class
    private int D = 2; // dimensionality
    private int K = 3; // number of classes

    private final List<Color> colors = List.of(Color.RED, Color.GREEN, Color.BLUE);

    public INDArray generateData() {
        INDArray X = Nd4j.zeros(K, D, N);// # data matrix (each row = single example)

        for (int j = 0; j < K; j++) {
            // r = np.linspace(0.0,1,N) # radius
            INDArray r = Nd4j.linspace(0.0, 1, N);// # radius

            // t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
            INDArray t = Nd4j
                    .linspace(j * 4, (j + 1) * 4, N)
                    .add(Nd4j.rand(new int[]{N})
                    .mul(0.2));// # theta

            // X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]

            X.getRow(j).putRow(0, r.mul(sin(t)));
            X.getRow(j).putRow(1, r.mul(cos(t)));

        }

        return X;
    }

    public   void visualizeData(INDArray X) {
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

    public static void main(String[] args) {
        Cs231s cs231s = new Cs231s();
        INDArray data = cs231s.generateData();
        cs231s.visualizeData(data);
    }
}