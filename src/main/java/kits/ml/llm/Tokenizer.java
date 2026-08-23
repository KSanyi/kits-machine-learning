package kits.ml.llm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.MatchResult;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class Tokenizer {

    private final Map<String, Integer> tokenToCode;
    private final Map<Integer, String> codeToToken;
    
    public Tokenizer(String text) {
        
        List<String> tokens = new ArrayList<>(tokenize(text));
        
        List<String> sortedTokens = tokens.stream().distinct().sorted().collect(Collectors.toList());
        
        System.out.println("Found " + tokens.size() + " tokens");
        System.out.println(tokens.stream().limit(30).toList());
        
        tokenToCode = new LinkedHashMap<>();
        codeToToken = new LinkedHashMap<>(); 
        int counter = 0;
        for(String token : sortedTokens) {
            tokenToCode.put(token, counter);
            codeToToken.put(counter, token);
            counter++;
        }
    }

    public List<Integer> encode(String text) {
        
        List<String> tokens = tokenize(text);
        
        List<Integer> codes = new ArrayList<>();
        for(String token : tokens) {
            codes.add(tokenToCode.get(token));
        }
        
        return codes;
    }
    
    List<String> tokenize(String text) {
        
        Pattern pattern = Pattern.compile("[^,.:;?_!\"()\\-'\\s]+|--|[,.:;?_!\"()'\\s]");
        
        return pattern.matcher(text)
                .results()
                .map(MatchResult::group)
                .filter(t -> !t.strip().isEmpty())
                .toList();
    }

    public String decode(List<Integer> codes) {
        String text = codes.stream().map(codeToToken::get).collect(Collectors.joining(" "));
        text = text.replaceAll("\\s+([,.:;?_!\\\"()'])", "$1");
        text = text.replaceAll("\n ", "\n");
        return text;
    }

}
