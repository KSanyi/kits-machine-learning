package kits.ml.digitsstanadlone;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FileUtil {

    private static final Logger log = LoggerFactory.getLogger(FileUtil.class);
    
    static List<LearningData> createLearningData(String path) throws Exception {
        
        log.info("Loading data from {}", path);
        List<LearningData> learningDatas = new ArrayList<>();
        for(File folder : Paths.get(path).toFile().listFiles()) {
            String name = folder.getName();
            int digit = Integer.parseInt(name);
            for(File file : folder.listFiles()) {
                Vector input = createInput(file);
                learningDatas.add(new LearningData(input, digit));
            }
            log.info("Data loaded for folder {}", name);
        }
        
        return learningDatas;
    }
    
    private static Vector createInput(File file) throws Exception {
        BufferedImage img = ImageIO.read(file);

        int width = img.getWidth();
        int height = img.getHeight();

        double[] data = new double[height * width];
        int index = 0;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = img.getRGB(x, y);
                int greyScale = convertToGreyScale(rgb);
                data[index++] = (greyScale / 255.0 - 0.5) / 2;
            }
        }
        
        return new Vector(data);
    }
    
    private static int convertToGreyScale(int rgb) {
        int r = (rgb >> 16) & 0xFF;
        int g = (rgb >> 8) & 0xFF;
        int b = rgb & 0xFF;
        return (r + g + b) / 3;
    }
    
}
