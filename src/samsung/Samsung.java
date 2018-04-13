/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package samsung;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Ian
 */
public class Samsung {
    
    public static int classIndex = 0;
    public static int count_no;
    public static int count_yes;
    public static int count_multi;
    public static Instances noArrayInstances;
    public static Instances yesArrayInstances;
    public static ArrayList<Instances> multiInstanceArray = new ArrayList<>(); 
    public static String t1="N0 [label=\"0";
    public static String t2="N0 [label=\"1";

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        
               
        DataSource source = new DataSource("docs/samsung.arff");
        Instances allusers=source.getDataSet();
        if (allusers.classIndex() == -1)
            classIndex=allusers.numAttributes()-1;
        allusers.setClassIndex(classIndex);
        //System.out.println(allusers.numInstances());
        noArrayInstances = new Instances(allusers, 0);
        yesArrayInstances = new Instances(allusers, 0);
        //multiArrayInstances = new Instances(allusers, 0);

        for (int i = 0; i < allusers.numInstances(); i = i + 12) {
            //int userID = (int)allusers.instance(i).value(0);
            Instances singleUserInstances = new Instances(allusers, i, 12);
            
            FilteredClassifier cls = wekaFunctions.train(singleUserInstances, classIndex); // train
//            double accuracy =  wekaFunctions.eval(cls, singleUserInstances,singleUserInstances); // eval
//            System.out.println("User #:" +userID);
//            System.out.println("Classifier :" +fc);
//            System.out.println("Accuracy :" +accuracy);

            //cls.graph() store this in a string and use string functions to parse it; put in if conditions to determine the clusters
            if(cls.graph().contains(t1)){
                count_no++;
                noArrayInstances = wekaFunctions.merge(noArrayInstances, singleUserInstances);
            }
            else if(cls.graph().contains(t2)){
                count_yes++;
                yesArrayInstances = wekaFunctions.merge(yesArrayInstances, singleUserInstances);
            }
            else {
                count_multi++;
                multiInstanceArray.add(singleUserInstances);

                //multiAccuracyList.add(wekaFunctions.eval(cls, singleUserInstances,singleUserInstances));               
            } 
        }
        System.out.println("Single 'NO' node trees: " + count_no);
        System.out.println("Single 'YES' node trees: " + count_yes);
        System.out.println("Multi-node trees: " + count_multi);         

        System.out.println("noArray's accuracy is: " + wekaFunctions.trainSelfEval(noArrayInstances));
        System.out.println("yesArray's accuracy is: " + wekaFunctions.trainSelfEval(yesArrayInstances));
        System.out.println(multiInstanceArray.size());
        System.out.println("===========================================");
        System.out.println("");
        for (int i = 0; i < 20; i++) {
            System.out.println("");
            System.out.println("/*****************************************/");
            System.out.println("/*                REPEAT "+(i+1)+"               */");
            System.out.println("/*****************************************/");
            System.out.println("");
            merge(434);
        }
    }
    
    public static void merge(int ROUNDS) throws Exception{
        ArrayList<Instances> instancesArray = new ArrayList<>(multiInstanceArray);
        Collections.shuffle(instancesArray);
        //System.out.println(instancesArray);
        for (int round = 0; round < ROUNDS; round++) {
            //double maxPairAccuracy = 0;
            double maxAccuracyDiff = -10000;
            int minTreeSize = 10000;
            int maxIndexLeftHand = -1;
            int maxIndexRightHand = -1;
            Instances maxPairInstance = null;
            int index=1;
            
            //Collections.shuffle(instancesArray);
            
            System.out.println("Round: " + (round+1) );
            for (int i = 0; i < instancesArray.size(); i++) {
                for (int j = i+1; j < instancesArray.size(); j++) {
                    Instances currentPairData = wekaFunctions.merge(instancesArray.get(i), instancesArray.get(j));

                    double currentPairAccuracy = wekaFunctions.trainSelfEval(currentPairData);
                    String cls = wekaFunctions.train(currentPairData, classIndex).getClassifier().toString();
                    int currentTreeSize = Integer.parseInt( cls.substring(cls.length()-3, cls.length()-1).replaceAll(".*[^\\d](?=(\\d+))","") );
                    double pre = (wekaFunctions.trainSelfEval(instancesArray.get(i))*instancesArray.get(i).numInstances()
                            + wekaFunctions.trainSelfEval(instancesArray.get(j))*instancesArray.get(j).numInstances())/12;

                    double after = (currentPairAccuracy*currentPairData.numInstances())/12;
                    double currentAccuracyDiff = after - pre;

//                    System.out.print((index++) + ": ");
//                    System.out.println("MaxDiff: " + maxAccuracyDiff);
//                    System.out.println("MinTree: " + minTreeSize);
//                    System.out.println("CurDiff: " + currentAccuracyDiff);
//                    System.out.println("CurTree: " + currentTreeSize);

                    if (currentAccuracyDiff > maxAccuracyDiff) {
                        maxAccuracyDiff = currentAccuracyDiff;
                        minTreeSize = currentTreeSize;
                        maxIndexLeftHand = i;
                        maxIndexRightHand = j;
                        maxPairInstance = currentPairData;
                    }
                    else if (currentAccuracyDiff == maxAccuracyDiff) {
                        if (currentTreeSize < minTreeSize) {
                            minTreeSize = currentTreeSize;
                            maxIndexLeftHand = i;
                            maxIndexRightHand = j;
                            maxPairInstance = currentPairData;                            
                        }
                    }
                }
            }

            instancesArray.set(maxIndexLeftHand, maxPairInstance);
            instancesArray.remove(maxIndexRightHand);
            System.out.println("Clusters size: " + instancesArray.size());
            System.out.println(maxIndexLeftHand + "," + maxIndexRightHand + ": " + maxAccuracyDiff);
            
            if (ROUNDS - round < 4) {
                System.out.println();                
                int totalTreeSize = 0;
                double totalAccuracy=0;
                for (int j = 0; j < instancesArray.size(); j++) {
                	System.out.println( "===========================");
                	System.out.println( "Cluster:" + (j+1));
                	System.out.println( "===========================");
                	// Output userid in each cluster
                	for (int i = 0; i < instancesArray.get(j).numInstances(); i = i + 12) {
                        System.out.print( (int)instancesArray.get(j).instance(i).value(0) + ", ");
                	}
                	System.out.println();
                	// number of users in the cluster
                    int numUsers = instancesArray.get(j).numInstances()/12;
                    
                    // classifier and tree size
                    FilteredClassifier cls = wekaFunctions.train(instancesArray.get(j));
                    String clsString = cls.getClassifier().toString();
                    System.out.println(clsString);
                    String str = clsString.substring(clsString.length()-4, clsString.length()-1).replaceAll(".*[^\\d](?=(\\d+))","");           
                    int treeSize = Integer.parseInt(str);
                    
                    // cluster accuracy
                    //double acc = wekaFunctions.eval(cls, instancesArray.get(j), instancesArray.get(j));
                    // Cross Validation
                    double acc = wekaFunctions.selfCVEval(instancesArray.get(j));
                    
                    System.out.println("numUsers: " + numUsers);
                    System.out.println("treeSize: " + treeSize); 
                    System.out.println("Accuracy: " + acc);
                    System.out.println("---------------------------------------");
                    System.out.println();
                    totalTreeSize += treeSize;          
                    totalAccuracy += acc*numUsers;
                }
                System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");               
                System.out.println("Total TreeSize: " + totalTreeSize);
                System.out.println("Total Accuracy: " + totalAccuracy);
                System.out.println("Avrge Accuracy: " + totalAccuracy/436);
                System.out.println("Total    Value: " + totalAccuracy/totalTreeSize);
            }            
            System.out.println("=======================================");
            System.out.println();
        }
    }
}
