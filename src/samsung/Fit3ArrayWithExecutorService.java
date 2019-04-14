package samsung;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import static samsung.wekaFunctions.eval;
import static samsung.wekaFunctions.merge;
import static samsung.wekaFunctions.trainWithOption;


import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSource;
 
/**
 * @author Crunchify.com
 * 
 */
 
public class Fit3ArrayWithExecutorService {
    private static final String cf = "0.01";
    private static final int MAXEXPTIMES = 50;
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
            instanceArray.add(new Instances(allusers, i, 12));
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
                new FileWriter("./docs/data/Fit3profile1_"+cf+".arff"))) {
            writer1.write(finalData1.toString());
            writer1.flush();
            writer1.close();
            CSVSaver s1 = new CSVSaver();
            s1.setFile(new File("./docs/data/Fit3profile1_"+cf+".csv"));
            s1.setInstances(finalData1);
            s1.setFieldSeparator(",");
            s1.writeBatch();
        }
        try (BufferedWriter writer2 = new BufferedWriter(
                new FileWriter("./docs/data/Fit3profile2_"+cf+".arff"))) {
            writer2.write(finalData2.toString());
            writer2.flush();
            writer2.close();
            CSVSaver s2 = new CSVSaver();
            s2.setFile(new File("./docs/data/Fit3profile2_"+cf+".csv"));
            s2.setInstances(finalData2);
            s2.setFieldSeparator(",");
            s2.writeBatch();
        }
        try (BufferedWriter writer3 = new BufferedWriter(
                new FileWriter("./docs/data/Fit3profile3_"+cf+".arff"))) {
            writer3.write(finalData3.toString());
            writer3.flush();
            writer3.close();
            CSVSaver s3 = new CSVSaver();
            s3.setFile(new File("./docs/data/Fit3profile3_"+cf+".csv"));
            s3.setInstances(finalData3);
            s3.setFieldSeparator(",");
            s3.writeBatch();
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
            double accuracy = 0;
            
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
                    fc1 = trainWithOption(data1, cf);
                    fc2 = trainWithOption(data2, cf);
                    fc3 = trainWithOption(data3, cf);
                }while ( fc1.numElements() == fc2.numElements() || fc1.numElements() == fc3.numElements() || fc2.numElements() == fc3.numElements());
                int expTimes = 0;
                for (expTimes = 0; expTimes < MAXEXPTIMES; expTimes++){
                    double Array1Accuracy = eval(fc1, data1, data1);
                    double Array2Accuracy = eval(fc2, data2, data2);
                    double Array3Accuracy = eval(fc3, data3, data3);
                    
                    System.out.println("Thread " + t.getId()
                            + " Iteration: " + expTimes
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

                        double accuracy1 = eval(fc1, data1Backup, user);
                        double accuracy2 = eval(fc2, data2Backup, user);
                        double accuracy3 = eval(fc3, data3Backup, user);
                        if (accuracy1 >= Math.max(accuracy2, accuracy3)){
                            data1 = merge(data1, user);
                        }
                        else if (accuracy2 > accuracy3)
                            data2 = merge(data2, user);
                        else if (accuracy3 > accuracy2)
                            data3 = merge(data3, user);
                        else if (accuracy2 == accuracy3)
                        {
                            Random randomNum = new Random();
                            if (randomNum.nextInt() % 2 == 0)
                                data2 = merge(data2, user);
                            else
                                data3 = merge(data3, user);
                        }
                    }
                    if ( data1.size()==data1Backup.size() && data2.size()==data2Backup.size() && lastAccuracy1 == Array1Accuracy && lastAccuracy2 == Array2Accuracy && lastAccuracy3==Array3Accuracy ){
                        accuracy = (data1.numInstances()/12*Array1Accuracy + data2.numInstances()/12*Array2Accuracy + data3.numInstances()/12*Array3Accuracy)/1133;
                        setMax(accuracy, data1, data2, data3);
                        break;
                    }
                    if (expTimes >= MAXEXPTIMES-1){
                        accuracy = (data1.numInstances()/12*Array1Accuracy + data2.numInstances()/12*Array2Accuracy + data3.numInstances()/12*Array3Accuracy)/1133;
                        setMax(accuracy, data1, data2, data3);
                    }
                    fc1 = trainWithOption(data1, cf);
                    fc2 = trainWithOption(data2, cf);
                    fc3 = trainWithOption(data3, cf);
                }
                
            } catch (Exception ex) {
                Logger.getLogger(Fit2ArrayWithExecutorService.class.getName()).log(Level.SEVERE, null, ex);
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
