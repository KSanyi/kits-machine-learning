package kits.ml.llm;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import kits.ml.util.FrequencyMap;
import kits.ml.util.Pair;

public class BPETokenizer {

    private final Map<String, Integer> tokenToTokenId = new HashMap<>();
    private final Map<Integer, String> tokenIdToToken = new HashMap<>();
    private final Map<Pair<Integer, Integer>, Integer> merges = new HashMap<>();
    
    public void train(String text, int vocabularySize) {
        
        StringBuilder processedText = new StringBuilder();
        for(int i=0;i<text.length();i++) {
            char ch = text.charAt(i);
            if(ch == ' ') {
                processedText.append("Ġ");
            } else {
                processedText.append(ch);
            }
        }
        
        for(int i=0;i<256;i++) {
            String ch = Character.toString((char)i);
            tokenToTokenId.put(ch, i);
            tokenIdToToken.put(i, ch);
        }
        
        List<Character> charsInText = processedText.chars().mapToObj(ch -> (char)ch).distinct().sorted().toList();
        
        for(char ch : charsInText) {
            if(!tokenToTokenId.containsKey(Character.toString(ch))) {
                int index = tokenToTokenId.size();
                tokenToTokenId.put(Character.toString(ch), index);
                tokenIdToToken.put(index, Character.toString(ch));
            }
        }
        
        List<Integer> tokenIds = processedText.chars().mapToObj(ch -> tokenToTokenId.get(Character.toString((char)ch))).toList();
        
        while(tokenToTokenId.size() < vocabularySize) {
            Pair<Integer, Integer> mostFrequentPair = findMostFrequentPair(tokenIds);
            int newIndex = tokenToTokenId.size();
            String string = tokenIdToToken.get(mostFrequentPair.first()) + tokenIdToToken.get(mostFrequentPair.second());
            tokenToTokenId.put(string, newIndex);
            tokenIdToToken.put(newIndex, string);
            merges.put(mostFrequentPair, newIndex);
            if(tokenToTokenId.size() % 100 == 0) System.out.println(tokenToTokenId.size());
            
            tokenIds = replacePairs(tokenIds, mostFrequentPair, newIndex);
        }
        
        tokenToTokenId.keySet().stream().sorted(Comparator.comparing(String::length).reversed()).limit(50).forEach(System.out::println);
    }

    private static List<Integer> replacePairs(List<Integer> tokenIds, Pair<Integer, Integer> pair, int newIndex) {
        List<Integer> updatedTokens = new ArrayList<>();
        int i=0;
        for(;i<tokenIds.size()-1;i++) {
            int first = tokenIds.get(i);
            int second = tokenIds.get(i+1);
            if(pair.first() == first && pair.second() == second) {
                updatedTokens.add(newIndex);
                i++;
            } else {
                updatedTokens.add(first);
            }
        }
        if(i == tokenIds.size()-1) {
            updatedTokens.add(tokenIds.get(i));
        }
        return updatedTokens;
    }

    private static Pair<Integer, Integer> findMostFrequentPair(List<Integer> tokenIds) {
        
        FrequencyMap<Pair<Integer, Integer>> frequencyMap = new FrequencyMap<>();
        for(int i=0;i<tokenIds.size()-1;i++) {
            int first = tokenIds.get(i);
            int second = tokenIds.get(i+1);
            frequencyMap.put(new Pair<>(first, second));
        }
        
        return frequencyMap.mostFrequent();
    }

    public List<Integer> encode(String text) {
        List<Integer> tokenIds = new ArrayList<Integer>();
        
        String[] words = text.split(" ");
        List<String> tokens = new ArrayList<String>();
        tokens.add(words[0]);
        for(int i=1;i<words.length;i++) {
            tokens.add("Ġ" + words[i]);
        }
        
        for(String token : tokens) {
            if(tokenToTokenId.containsKey(token)) {
                tokenIds.add(tokenToTokenId.get(token));
            } else {
                tokenIds.addAll(tokenize(token));
            }
        }
        
        return tokenIds;
    }
    
    private List<Integer> tokenize(String word) {
        List<Integer> tokenIds = word.chars().mapToObj(ch -> tokenToTokenId.get(Character.toString(ch))).toList();
        
        boolean hasMerge = true;
        while(hasMerge) {
            hasMerge = false;
            for(int i=0;i<tokenIds.size()-1;i++) {
                int tokenId1 = tokenIds.get(i);
                int tokenId2 = tokenIds.get(i+1);
                Pair<Integer, Integer> tokenIdPair = new Pair<>(tokenId1, tokenId2);
                Integer newTokenId = merges.get(tokenIdPair);
                if(newTokenId != null) {
                    tokenIds = replacePairs(tokenIds, tokenIdPair, newTokenId);
                    hasMerge = true;
                    break;
                }
            }
        }
        
        return tokenIds;
    }

    public String decode(List<Integer> tokenIds) {
        StringBuilder decodedBuilder = new StringBuilder();
        for(int tokenId : tokenIds) {
            String token = tokenIdToToken.get(tokenId);
            if(token.startsWith("Ġ")) {
                decodedBuilder.append(" " + token.substring(1));
            } else {
                decodedBuilder.append(token);    
            }
        }
        return decodedBuilder.toString();
    }

}
