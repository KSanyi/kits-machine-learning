package kits.ml.util;

public class Logger {

    private static final long start = System.currentTimeMillis();
    
    public static void log(String message) {
        long now = System.currentTimeMillis();
        long elapsedSec = (now - start) / 1000;
        long minutes = elapsedSec / 60;
        long hours = minutes / 60;
        long remainingMinutes = minutes % 60;
        String minutesString = (remainingMinutes < 10 ? "0" : "") + remainingMinutes;
        long remainingSec = elapsedSec % 60;
        String secString = (remainingSec < 10 ? "0" : "") + remainingSec;
        System.out.println(hours + ":" + minutesString + ":" + secString + ": " + message);
    }
    
}

