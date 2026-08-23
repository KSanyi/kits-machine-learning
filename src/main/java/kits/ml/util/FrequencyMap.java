package kits.ml.util;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Collectors;

public class FrequencyMap<T> {

    private final Map<T, Long> frequencyMap = new HashMap<>();
    
    public void put(T elem) {
        frequencyMap.merge(elem, 1L, Long::sum);
    }
    
    public long frequency(T object) {
        return frequencyMap.getOrDefault(object, 0L);
    }
    
    public T mostFrequent() {
        return frequencyMap.entrySet()
                .stream()
                .max(Map.Entry.comparingByValue())
                .map(e -> e.getKey()).orElse(null);
    }
    
    @Override
    public String toString() {
        Map<T, Long> sortedByFrequency = frequencyMap.entrySet()
                .stream()
                .sorted(Map.Entry.comparingByValue())
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));
        
        return sortedByFrequency.toString();
    }
    
}
