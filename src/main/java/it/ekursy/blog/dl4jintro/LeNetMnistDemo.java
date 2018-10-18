package it.ekursy.blog.dl4jintro;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import spark.Request;

import javax.imageio.ImageIO;
import javax.servlet.MultipartConfigElement;
import javax.servlet.ServletException;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

import static spark.Spark.*;

public class HelloWorld {
    public static void main(String[] args) throws Exception {
        MultiLayerNetwork net = MultiLayerNetwork.load(new File("src/main/resources/models/lenet-mnist-model.zip"), false);

        staticFiles.location("/static/");
        staticFiles.expireTime(1);

        File uploadDir = new File("upload");
        uploadDir.mkdir();

        staticFiles.externalLocation("upload");

        get("/hello", (req, res) -> "Hello World");

        post("/mnist", (req, res) -> {
            res.type("application/json");

            Path tempFile = receiveUploadedFile(uploadDir, req);

            try {
                BufferedImage inputImage = ImageIO.read(tempFile.toFile());

                BufferedImage image = invertColors(inputImage);

                BufferedImage gray = resize(image);

                INDArray digit = toINDArray(gray);

                INDArray output = net.output(digit);

                System.out.println(output);
                double max = output.getRow(0).max().getDouble(0);
                if (max > 0.30) {
                    int idx = findMatchingIndex(output, max);
                    return "{\"digit\":\"" + idx + "\", \"score\": " + max + "}";
                } else {
                    res.status(404);
                    return "{}";
                }
            } catch (Exception e) {
                e.printStackTrace();
                res.status(500);
                return "{}";
            } finally {
                Files.delete(tempFile);
            }
        });

    }

    @NotNull
    private static Path receiveUploadedFile(File uploadDir, Request req) throws IOException, ServletException {
        Path tempFile = Files.createTempFile(uploadDir.toPath(), "", "");

        req.attribute("org.eclipse.jetty.multipartConfig", new MultipartConfigElement("/temp"));

        try (InputStream input = req.raw().getPart("uploaded_file").getInputStream()) { // getPart needs to use same "name" as input field in form
            Files.copy(input, tempFile, StandardCopyOption.REPLACE_EXISTING);
        }
        return tempFile;
    }

    private static int findMatchingIndex(INDArray output, double max) {
        for (int i = 0; i < 10; i++) {
            if (max == output.getRow(0).getDouble(i)) {
                return i;
            }
        }
        return -1;
    }

    private static INDArray toINDArray(BufferedImage gray) {
        INDArray digit = Nd4j.create(28, 28);
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                Color c = new Color(gray.getRGB(i, j));
                digit.putScalar(new int[]{j, i}, (c.getGreen() & 0xFF));
            }
        }

        return digit.reshape(1, 1, 28, 28).divi(0xff);
    }

    @NotNull
    private static BufferedImage resize(BufferedImage image) {
        BufferedImage gray = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);

        Graphics2D g = (Graphics2D) gray.getGraphics();
        g.setBackground(Color.WHITE);
        g.clearRect(0, 0, 28, 28);
        g.drawImage(image.getScaledInstance(28, 28, Image.SCALE_SMOOTH), 0, 0, null);
        g.dispose();
        return gray;
    }

    @NotNull
    private static BufferedImage invertColors(BufferedImage inputImage) {
        BufferedImage gray = new BufferedImage(inputImage.getWidth(), inputImage.getHeight(), BufferedImage.TYPE_BYTE_GRAY);

        Graphics2D g = (Graphics2D) gray.getGraphics();
        g.setBackground(Color.WHITE);
        g.clearRect(0, 0, inputImage.getWidth(), inputImage.getHeight());
        g.drawImage(inputImage, 0, 0, null);
        g.dispose();

        for (int x = 0; x < gray.getWidth(); x++) {
            for (int y = 0; y < gray.getHeight(); y++) {
                int rgba = gray.getRGB(x, y);
                Color col = new Color(rgba, true);
                col = new Color(255 - col.getRed(),
                        255 - col.getGreen(),
                        255 - col.getBlue());
                gray.setRGB(x, y, col.getRGB());
            }
        }
        return gray;
    }
}