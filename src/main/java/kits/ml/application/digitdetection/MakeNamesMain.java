package kits.ml.application.digitdetection;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

public class MakeNamesMain {

    public static void main(String[] args) throws IOException {
        
        List<String> names = Files.readAllLines(Paths.get("input/names.txt"));

        Set<Character> chars = collectChars(names);
        chars.add('>');
        chars.add('.');
        
        NameGenerator nameGenerator = new NameGenerator(chars);
        
        for(String name : names) {
            nameGenerator.add(">", name.charAt(0));
            for(int i=1;i<name.length();i++) {
                nameGenerator.add(name.substring(i-1, i), name.charAt(i));
            }
//            for(int i=2;i<name.length();i++) {
//                nameGenerator.add(name.substring(i-2, i), name.charAt(i));
//            }
            nameGenerator.add(name.substring(name.length()-1, name.length()), '.');
//            nameGenerator.add(name.substring(name.length()-2, name.length()), '.');
        }
        
        nameGenerator.initSamplers();
        
        for(int i=0;i<10;i++) {
            System.out.println(nameGenerator.generate());
        }
        
        double avgCost = 0;
        int n=0;
        for(String name : names) {
            avgCost += nameGenerator.calculateLogLikelihood(name);
            n += name.length()+1;
        }
        avgCost /= n;
        
        System.out.println(avgCost);
        
//        System.out.println(nameGenerator.calculateLogLikelihood("emma"));
//        System.out.println(nameGenerator.calculateLogLikelihood("olivia"));
//        System.out.println(nameGenerator.calculateLogLikelihood("ava"));
//        
//        System.out.println(nameGenerator.calculateLogLikelihood("xwqf"));
        
    }
    
    private static Set<Character> collectChars(List<String> names) {
        Set<Character> chars = new HashSet<>();
        for(String name : names) {
            name.chars().forEach(c -> chars.add((char)c));
        }
        return chars;
    }
    
    static class NameGenerator {
        
        private final Set<Character> chars;
        private final Map<String, FrequencyMap<Character>> frequenciesMap = new HashMap<>();
        private final Map<String, Sampler> samplerMap = new HashMap<>();

        public NameGenerator(Set<Character> chars) {
            this.chars = chars;
            for(char ch1 : chars) {
                chars.forEach(ch -> frequenciesMap.put(""+ch1, new FrequencyMap<Character>()));
//                for(char ch2 : chars) {
//                    chars.forEach(ch -> frequenciesMap.put(""+ch1+ch2, new FrequencyMap<Character>()));
//                }   
            }
        }

        public String generate() {
            StringBuilder sb = new StringBuilder(">");
            char current = '>';
            while(current != '.') {
                String prefix = sb.substring(Math.max(0, sb.length()-1), sb.length());
                current = samplerMap.get(prefix).sample();
                if(current == '.' && sb.length() < 3) {
                    current = '>';
                    continue;
                }
                sb.append(current);
            }
            return sb.substring(0, sb.length()-1).substring(1);
        }

        public void add(String string, char follow) {
            frequenciesMap.get(string).put(follow);
        }
        
        public void initSamplers() {
            for(String string : frequenciesMap.keySet()) {
                samplerMap.put(string, new Sampler(chars, frequenciesMap.get(string), '>'));
            }
        }
        
        public double calculateLogLikelihood(String name) {
            double logSum = 0;
            name = ">" + name;
            for(int i=0;i<name.length()-1;i++) {
                String prefix = name.substring(i, i+1);
                logSum += Math.log(samplerMap.get(prefix).probability(name.charAt(i+1)));
                System.out.println(prefix + " -> " + " "+ name.charAt(i+1) + ": " + samplerMap.get(prefix).probability(name.charAt(i+1)));
            }
            String prefix = name.substring(name.length()-1);
            logSum += Math.log(samplerMap.get(prefix).probability('.'));
            System.out.println(prefix + " -> " + " .: " + samplerMap.get(prefix).probability('.'));
            
            return -logSum;
        }
    }
    
    static class FrequencyMap<T> {

        private Map<T, Integer> map = new HashMap<>();

        public void put(T elem) {
            map.put(elem, map.getOrDefault(elem, 0) + 1);
        }

        public Integer frequency(T elem) {
            return map.getOrDefault(elem, 0);
        }

        @Override
        public String toString() {
            return map.keySet().stream().map(elem -> elem + " " + map.get(elem)).collect(Collectors.joining("\n"));
        }

    }
    
    static class Sampler {
        
        private final Random random;
        private final Set<Character> chars;
        private final Map<Character, Double> probabilityMap;
        
        public Sampler(Set<Character> chars, FrequencyMap<Character> frequencyMap, char starter) {
            this(chars, frequencyMap, starter, new Random().nextInt(10000000));
        }
        
        public double probability(char ch) {
            return probabilityMap.get(ch);
        }

        public Sampler(Set<Character> chars, FrequencyMap<Character> frequencyMap, char starter, long seed) {
            this.chars = chars;
            random = new Random(seed);
            double sum = chars.stream().filter(ch -> ch != starter).mapToDouble(ch -> max(0.1, frequencyMap.frequency(ch))).sum();
            probabilityMap = chars.stream().collect(Collectors.toMap(ch -> ch, ch -> max(0.1, frequencyMap.frequency(ch)) / sum));
        }

        public char sample() {
            double randomNumber = random.nextDouble();
            double runningSum = 0;
            for(char ch : chars) {
                runningSum += probabilityMap.get(ch);
                if(randomNumber < runningSum) {
                    return ch;
                }
            }
            throw new IllegalStateException();
        }
        
    }
    
    private static double max(double a, double b) {
        return a > b ? a : b;
    }

}
