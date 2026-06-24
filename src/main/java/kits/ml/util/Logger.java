package kits.ml.util;

public class Logger {

    private final long start = System.currentTimeMillis();
    
    public void log(String message) {
        long now = System.currentTimeMillis();
        long elapsedSec = (now - start) / 1000;
        System.out.println(elapsedSec + " sec: " + message);
    }
    
}

