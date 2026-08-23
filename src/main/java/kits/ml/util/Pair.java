package kits.ml.util;

public record Pair<S, T>(S first, T second) {

    @Override
    public String toString() {
        return "(" + first + ", " + second + ")";
    }
    
}
