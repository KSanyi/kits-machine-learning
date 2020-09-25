package kits.ml.util;

public class StopWatch {

    public static void timed(Runnable runnable) {
        
        long start = System.currentTimeMillis();
        try {
            runnable.run();
        } catch(Throwable th) {
            long stop = System.currentTimeMillis();
            System.out.println("Failed with error after " + format(start, stop) + " " + th);
        }
        long stop = System.currentTimeMillis();
        System.out.println("Calculation took " + format(start, stop));
    }
    
    private static String format(long start, long end) {
        long diffInMillis = end - start;
        return diffInMillis > 10_000 ? (diffInMillis / 1000 + " seconds") : (diffInMillis + " millis");
    }
    
}
