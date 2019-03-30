package kits.ml.core;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class FileReader {

    private final static String DELIMETER = ",";

    public static Input readInput(String string) {
        // TODO Auto-generated method stub
        return null;
    }

    public static List<LearningData> readLearningDataSet(String filePath) {
        try {
            List<String> lines = Files.readAllLines(Paths.get(filePath));
            return lines.stream().map(FileReader::createLearningData).collect(Collectors.toList());
        } catch (IOException ex) {
            throw new RuntimeException("Can not read file: " + filePath);
        }
    }

    private static LearningData createLearningData(String line) {
        String[] parts = line.split(DELIMETER);
        double[] values = Stream.of(parts).mapToDouble(Double::parseDouble).toArray();
        double[] inputValues = Arrays.copyOfRange(values, 0, values.length - 1);
        double output = values[values.length - 1];
        return new LearningData(new Input(inputValues), output);
    }

}
