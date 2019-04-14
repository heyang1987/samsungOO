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
 
public class Fit2ArrayWithExecutorServiceMLP {
    private static final int MYTHREADS = 50;
    public static Instances allusers;
    public static double maxCorrectPercentage = 0;
    public static Instances finalData1;
    public static Instances finalData2;
 
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
			System.out.println("Round "+(i+1)+":");
			Runnable worker = new MyRunnable(instanceArray);
			executor.execute(worker);
		}
		executor.shutdown();
		// Wait until all threads are finish
		while (!executor.isTerminated()) {
 
		}
		System.out.println("\nFinished all threads");
        try (BufferedWriter writer1 = new BufferedWriter(new FileWriter("./docs/data/MLPFit2profile1.arff"))) {
            writer1.write(finalData1.toString());
            writer1.flush();
            writer1.close();
        }
        try (BufferedWriter writer2 = new BufferedWriter(new FileWriter("./docs/data/MLPFit2profile2.arff"))) {
            writer2.write(finalData2.toString());
            writer2.flush();
            writer2.close();
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
            Instances data1 = new Instances(allusers,0);
            Instances data2 = new Instances(allusers,0);
            FilteredClassifier fc1;
            FilteredClassifier fc2;
            
            try {
                do{
                    Collections.shuffle(instanceArray);
                    //System.out.println(instanceArray);
                    for (int i = 0; i < Math.round(instanceArray.size()/2); i++) {
                        data1 = merge(data1, instanceArray.get(i));
                    }
                    for (int j = Math.round(instanceArray.size()/2); j < instanceArray.size(); j++) {
                        data2 = merge(data2, instanceArray.get(j));
                    }
                    fc1 = mlpCls(data1);
                    //System.out.println(fc1);
                    fc2 = mlpCls(data2);
                    //System.out.println(fc2);
                    
                }while ( fc1 == fc2 );
                
                for (int expTimes = 0; expTimes < 200; expTimes++){
                    double Array1Accuracy = eval(fc1, data1, data1);
                    double Array2Accuracy = eval(fc2, data2, data2);
                    System.out.println("Thread "+ t.getId() +" Iteration: " + expTimes);
                    Instances data1Backup = new Instances(data1);  //BACKUPS HAVE BEEN MADE JUST IN CASE WE NEED TO PUT IT BACK IN THE SAME ARRAY
                    Instances data2Backup = new Instances(data2);
                    lastAccuracy1 = Array1Accuracy;
                    lastAccuracy2 = Array2Accuracy;
                    data1.clear();
                    data2.clear();
	
                    for (int i = 0; i < allusers.numInstances(); i = i + 12) {
                        Instances user = new Instances(allusers, i, 12);
                        int userID = (int)allusers.instance(i).value(0);

                        double accuracy1 = eval(fc1, data1Backup, user);
                        double accuracy2 = eval(fc2, data2Backup, user);
                        if (accuracy1 > accuracy2){
                            data1 = merge(data1, user);
                        }
                        else if (accuracy1 < accuracy2) {
                            data2 = merge(data2, user);
                        }
                        else if (accuracy1 == accuracy2) {
                            if (data1Backup.contains(user.instance(0))){
                                data1 = merge(data1, user);
                            }
                            else
                                data2 = merge(data2, user);
                        }
                    }
                    // System.out.println("data1: "+data1.size()/12);
                    if ( data1.size()==data1Backup.size() && lastAccuracy1 == Array1Accuracy && lastAccuracy2 == Array2Accuracy){
                        double accuracy = (data1.numInstances()*Array1Accuracy + data2.numInstances()*Array2Accuracy)/(data1.numInstances()+data2.numInstances());
                        //System.out.println("*****************************************************************************");
                        //System.out.println("Arrays converged within " + expTimes + " iterations");
                        //System.out.println("Cur Accuracy: " +  accuracy);
                        setMax(accuracy, data1, data2);
                        //System.out.println("*****************************************************************************");
//                        if (accuracy > maxCorrectPercentage){
//                            maxCorrectPercentage = accuracy;
//                            maxFc1 = fc1;
//                            maxFc2 = fc2;
//                            finalArray1Size = data1.numInstances()/12;
//                            finalArray2Size = data2.numInstances()/12;
//                            finalAccuracy1 = Array1Accuracy;
//                            finalAccuracy2 = Array2Accuracy;
//                                                finalData1 = data1;
//                                                finalData2 = data2;
//                        }
//                        System.out.println("Max Accuracy: " +  maxCorrectPercentage);
                        break;
                    }

                    fc1 = mlpCls(data1);
                    fc2 = mlpCls(data2);
                }
                
            } catch (Exception ex) {
                Logger.getLogger(Fit2ArrayWithExecutorServiceMLP.class.getName()).log(Level.SEVERE, null, ex);
            }
		}
	}
    
    public static synchronized void setMax(double accuracy, Instances data1, Instances data2)
    { 
        if (accuracy > maxCorrectPercentage){
            maxCorrectPercentage = accuracy;
            finalData1 = data1;
            finalData2 = data2;
        }
        System.out.println("Current Accuracy: " +  accuracy + ", Max Accuracy: " +  maxCorrectPercentage);
    }
}
