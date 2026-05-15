package kits.ml.util;

import java.util.function.Supplier;

public class StopWatch {

    public static void timed(Runnable runnable, String name) {
        long start = System.currentTimeMillis();
        runnable.run();
        long end = System.currentTimeMillis();
        System.out.println(name + " took " + format(start, end));
    }
    
    public static <T> T timed(Supplier<T> supplier, String name) {
        long start = System.currentTimeMillis();
        T result = supplier.get();
        long end = System.currentTimeMillis();
        
        System.out.println(name + " took " + format(start, end));
        
        return result;
    }
    
    private static String format(long start, long end) {
        long diffInMillis = end - start;
        return diffInMillis > 10_000 ? (diffInMillis / 1000 + " seconds") : (diffInMillis + " millis");
    }
    
}
