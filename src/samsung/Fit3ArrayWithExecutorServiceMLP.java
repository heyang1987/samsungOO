package samsung;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.logging.Level;
import java.util.logging.Logger;
import static samsung.wekaFunctions.eval;
import static samsung.wekaFunctions.merge;
import static samsung.wekaFunctions.mlpCls;


import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
 
/**
 * @author Crunchify.com
 * 
 */
 
public class Fit3ArrayWithExecutorServiceMLP {
    private static final int MYTHREADS = 48;
    private static Instances allusers;
    private static double maxCorrectPercentage = 0;
    private static Instances finalData1;
    private static Instances finalData2;
    private static Instances finalData3;
    private static int cnt = 0;
 
    public static void main(String args[]) throws Exception {
		ExecutorService executor = Executors.newFixedThreadPool(MYTHREADS);
		DataSource source = new DataSource("docs/samsung.arff");
        allusers=source.getDataSet();
        //System.out.println(allusers);
        if (allusers.classIndex() == -1)
            allusers.setClassIndex(allusers.numAttributes()-1);
        //System.out.println(mlp10foldCV(allusers));
        //System.out.println(trainSelfCVEval(allusers));
        // System.out.println(trainSelfEval(allusers)); // No Cross-validation
        
        
        ArrayList<Instances> instanceArray = new ArrayList<>();
        for (int i = 0; i < allusers.numInstances(); i = i + 12) {
            Instances singleUserInstances = new Instances(allusers, i, 12);
            instanceArray.add(singleUserInstances);
        }
        
		for (int i = 0; i < 200; i++) {
			System.out.println("Round "+(i+1)+" started:");
			Runnable worker = new MyRunnable(instanceArray);
			executor.execute(worker);
		}
		executor.shutdown();
		// Wait until all threads are finish
		while (!executor.isTerminated()) {
 
		}
		System.out.println("\nFinished all threads");
        try (BufferedWriter writer1 = new BufferedWriter(
                new FileWriter("./docs/data/MLPFit3profile1.arff"))) {
            writer1.write(finalData1.toString());
            writer1.flush();
            writer1.close();
        }
        try (BufferedWriter writer2 = new BufferedWriter(
                new FileWriter("./docs/data/MLPFit3profile2.arff"))) {
            writer2.write(finalData2.toString());
            writer2.flush();
            writer2.close();
        }
        try (BufferedWriter writer3 = new BufferedWriter(
                new FileWriter("./docs/data/MLPFit3profile3.arff"))) {
            writer3.write(finalData3.toString());
            writer3.flush();
            writer3.close();
        }
	}
 
	public static class MyRunnable implements Runnable {
		private final ArrayList<Instances> instanceArray;
 
		MyRunnable(ArrayList<Instances> instanceArray) {
			this.instanceArray = instanceArray;
		}
 
		@Override
		public void run() {
            Thread t = Thread.currentThread();
            double lastAccuracy1;
            double lastAccuracy2;
            double lastAccuracy3;
            Instances data1 = new Instances(allusers,0);
            Instances data2 = new Instances(allusers,0);
            Instances data3 = new Instances(allusers,0);
            FilteredClassifier fc1;
            FilteredClassifier fc2;
            FilteredClassifier fc3;
            
            try {
                do{
                    Collections.shuffle(instanceArray);
                    //System.out.println(instanceArray);
                    for (int i = 0; i < Math.round(instanceArray.size()/3); i++) {
                        data1 = merge(data1, instanceArray.get(i));
                    }
                    for (int j = Math.round(instanceArray.size()/3); j < Math.round(instanceArray.size()*2/3); j++) {
                        data2 = merge(data2, instanceArray.get(j));
                    }
                    for (int k = Math.round(instanceArray.size()*2/3); k < instanceArray.size(); k++) {
                        data3 = merge(data3, instanceArray.get(k));
                    }
                    fc1 = mlpCls(data1);
                    //System.out.println(fc1);
                    fc2 = mlpCls(data2);
                    //System.out.println(fc2);
                    fc3 = mlpCls(data3);
                }while ( fc1 == fc2 || fc2 == fc3 || fc1 == fc3);
                
                for (int expTimes = 0; expTimes < 200; expTimes++){
                    double Array1Accuracy = eval(fc1, data1, data1);
                    double Array2Accuracy = eval(fc2, data2, data2);
                    double Array3Accuracy = eval(fc3, data3, data3);
                    System.out.println("Thread "
                            + t.getId()
                            + " Iteration: " 
                            + expTimes
                    );
                    // BACKUPS HAVE BEEN MADE JUST IN CASE WE NEED 
                    // TO PUT IT BACK IN THE SAME ARRAY
                    Instances data1Backup = new Instances(data1);  
                    Instances data2Backup = new Instances(data2);
                    Instances data3Backup = new Instances(data3);
                  
                    lastAccuracy1 = Array1Accuracy;
                    lastAccuracy2 = Array2Accuracy;
                    lastAccuracy3 = Array3Accuracy;
                    
                    data1.clear();
                    data2.clear();
                    data3.clear();
	
                    for (int i = 0; i < allusers.numInstances(); i = i + 12) {
                        Instances user = new Instances(allusers, i, 12);
                        //int userId = (int)user.instance(0).value(0);
                        double[] accuracyArray = new double[5];
                        accuracyArray[0] = eval(fc1, data1Backup, user);
                        accuracyArray[1] = eval(fc2, data2Backup, user);
                        accuracyArray[2] = eval(fc3, data3Backup, user);
                        double max = accuracyArray[0];
                        int index = 0;
                        for(int j = 0; j < 3; j++)
                        {
                            if(max < accuracyArray[j])
                            {
                                max = accuracyArray[j];
                                index = j;
                            }
                        }
                        
                        if (data1Backup.contains(user.instance(0)) 
                                && accuracyArray[0] == max) {
                            data1 = merge(data1, user);              		
                        }
                        else if (data2Backup.contains(user.instance(0)) 
                                && accuracyArray[1] == max) {
                            data2 = merge(data2, user);              		
                        }
                        else if (data3Backup.contains(user.instance(0)) 
                                && accuracyArray[2] == max) {
                            data3 = merge(data3, user);              		
                        }
                        else {
                            switch (index) {
                                case 0:
                                    data1 = merge(data1, user); 
                                    break;
                                case 1:
                                    data2 = merge(data2, user); 
                                    break;
                                case 2:
                                    data3 = merge(data3, user); 
                                    break;
                                default:
                                    System.out.println("Switch Error!");
                                    break;
                            }
                        }
//                        if (accuracy1 > Math.max(accuracy2, accuracy3)){
//                            data1 = merge(data1, user);
//                        }
//                        else if (accuracy2 > accuracy3)
//                            data2 = merge(data2, user);
//                        else if (accuracy3 > accuracy2)
//                            data3 = merge(data3, user);
//                        else if (accuracy2 == accuracy3)
//                        {
//                            Random randomNum = new Random();
//                            if (randomNum.nextInt() % 2 == 0)
//                                data2 = merge(data2, user);
//                            else
//                                data3 = merge(data3, user);
//                        }
                    }
                    // System.out.println("data1: "+data1.size()/12);
                    if ( data1.size()==data1Backup.size() && data2.size()==data2Backup.size() && lastAccuracy1 == Array1Accuracy && lastAccuracy2 == Array2Accuracy && lastAccuracy3==Array3Accuracy ){
                        double accuracy = (data1.numInstances()/12*Array1Accuracy + data2.numInstances()/12*Array2Accuracy + data3.numInstances()/12*Array3Accuracy)/1133;
                        setMax(accuracy, data1, data2, data3);
                        break;
                    }

                    fc1 = mlpCls(data1);
                    fc2 = mlpCls(data2);
                    fc3 = mlpCls(data3);
                }
                
            } catch (Exception ex) {
                Logger.getLogger(Fit2ArrayWithExecutorServiceMLP.class.getName()).log(Level.SEVERE, null, ex);
            }
		}
	}
    
    public static synchronized void setMax(double accuracy, Instances data1, Instances data2, Instances data3)
    { 
        if (accuracy > maxCorrectPercentage){
            maxCorrectPercentage = accuracy;
            finalData1 = data1;
            finalData2 = data2;
            finalData3 = data3;
        }
        System.out.println((cnt++)+" Current Accuracy: " +  accuracy + ", Max Accuracy: " +  maxCorrectPercentage);
    }
}
