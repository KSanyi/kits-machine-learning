package kits.ml.core.math.linalg;

public class CholeskyDecompositionCalculator {

    public static Matrix calculate(Matrix A) {
        
        int n = A.getNrRows();
        
        double[][] L = new double[n][n];
        
        for (int i = 0; i < n; i++) {
            double[] rowI = L[i];
            for (int j = 0; j <= i; j++) {
                double[] rowJ = L[j];
                double sum = 0;
                for (int k = 0; k < j; k++) {
                    sum += rowI[k] * rowJ[k];
                }

                if (i == j) {
                    rowI[j] = Math.sqrt(A.values[i][i] - sum);
                } else {
                    rowI[j] = (1.0 / rowJ[j] * (A.values[i][j] - sum));
                }
            }
        }
        
        return new Matrix(L);
    }
    
    public static Matrix calculateAlex(Matrix A) {
        
        int n = A.getNrRows();
        
        double[][] L = new double[n][n];
        
        for (int i = 0; i < n; i++) {
            double[] aRowI = A.values[i];
            double sum1 = aRowI[i];
            double[] lRowI = L[i];
            for (int j = 0; j <= i; j++) {
                double x = lRowI[j];
                sum1 -= x * x;
            }
            
            if(sum1 > 0.000001) {
                
                double x = Math.sqrt(sum1);
                lRowI[i] = x;
                
                for(int j = i+1; j < n;j++) {
                    double sum2 = aRowI[j];
                    double[] lRowj = L[j];
                    for (int k = 0; k < i; k++) {
                        sum2 -= lRowI[k] * lRowj[k];
                    }
                    
                    lRowj[i] = sum2/x;
                }
            }
        }
        
        return new Matrix(L);
    }
    
    public static Matrix calculate2(Matrix A) {
        
        int n = A.getNrRows();
        
        double[][] L = new double[n][n];
        
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int k = 0; k < j; k++) {
                sum += L[j][k] * L[j][k];
            }
            L[j][j] = Math.sqrt(A.values[j][j] - sum);

            for (int i = j + 1; i < n; i++) {
                sum = 0;
                for (int k = 0; k < j; k++) {
                    sum += L[i][k] * L[j][k];
                }
                L[i][j] = (1.0 / L[j][j] * (A.values[i][j] - sum));
            }
        }
        
        return new Matrix(L);
    }
    
}
