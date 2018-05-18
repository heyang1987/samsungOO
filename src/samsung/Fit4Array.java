package samsung;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;

public class Fit4Array {
	
	public static ArrayList<Instances> instanceArray = new ArrayList<>();
	public static Instances data1;
	public static Instances data2;
	public static Instances data3;
	public static Instances data4;
	public static double maxCorrectPercentage = 0;
	public static FilteredClassifier fc1 = new FilteredClassifier();
	public static FilteredClassifier fc2 = new FilteredClassifier();
	public static FilteredClassifier fc3 = new FilteredClassifier();
	public static FilteredClassifier fc4 = new FilteredClassifier();
	public static FilteredClassifier maxFc1 = new FilteredClassifier();
	public static FilteredClassifier maxFc2 = new FilteredClassifier();
	public static FilteredClassifier maxFc3 = new FilteredClassifier();
	public static FilteredClassifier maxFc4 = new FilteredClassifier();
	public static int finalArray1Size = 0;
	public static int finalArray2Size = 0;
	public static int finalArray3Size = 0;
	public static int finalArray4Size = 0;
	public static double finalAccuracy1 = 0;
	public static double finalAccuracy2 = 0;
	public static double finalAccuracy3 = 0;
	public static double finalAccuracy4 = 0;
	public static Instances allusers;
	
	public static void main(String[] args) throws Exception {
		String[] cf;
		cf = new String[25];
		cf[0] = "0.01";
		cf[1] = "0.02";
		cf[2] = "0.03";
		cf[3] = "0.04";
		cf[4] = "0.05";
		cf[5] = "0.06";
		cf[6] = "0.07";
		cf[7] = "0.08";
		cf[8] = "0.09";
		cf[9] = "0.10";
		cf[10] = "0.11";
		cf[11] = "0.12";
		cf[12] = "0.13";
		cf[13] = "0.14";
		cf[14] = "0.15";
		cf[15] = "0.16";
		cf[16] = "0.17";
		cf[17] = "0.18";
		cf[18] = "0.19";
		cf[19] = "0.20";
		cf[20] = "0.21";
		cf[21] = "0.22";
		cf[22] = "0.23";
		cf[23] = "0.24";
		cf[24] = "0.25";
		
        DataSource source = new DataSource("samsung.arff");
        allusers=source.getDataSet();
        if (allusers.classIndex() == -1)
            allusers.setClassIndex(allusers.numAttributes()-1);
        //System.out.println(allusers.numInstances());

        for (int i = 0; i < allusers.numInstances(); i = i + 12) {
            Instances singleUserInstances = new Instances(allusers, i, 12);
            instanceArray.add(singleUserInstances);
        }
        
        int i = 0;
        for (i=0; i<25; i++) {
        	String strFilename = "./4_"+cf[i]+".txt";
        	System.out.println(strFilename);
            try {
            	File fileText = new File(strFilename);    
    	    	FileWriter fileWriter = new FileWriter(fileText);
				for (int j = 0; j < 1000; j++){
		            System.out.println("Round "+(j+1)+":");
		            converge(cf[i]);
				}
				System.out.println("*****************************************************************************");
				System.out.println(cf[i]+ "\tFinal Statistics:\n");
				System.out.println("maxfc1:\n"+maxFc1+"maxfc2:\n"+maxFc2+"maxfc3:\n"+maxFc3+"maxfc4:\n"+maxFc4);
				System.out.println("Array1's size: " +  finalArray1Size + "\t" + "accuracy: " + finalAccuracy1);
				System.out.println("Array2's size: " +  finalArray2Size + "\t" + "accuracy: " + finalAccuracy2);
				System.out.println("Array3's size: " +  finalArray3Size + "\t" + "accuracy: " + finalAccuracy3);
				System.out.println("Array4's size: " +  finalArray4Size + "\t" + "accuracy: " + finalAccuracy4);
				System.out.println("Max Correct Percentage: " +  maxCorrectPercentage);
				System.out.println("*****************************************************************************");
				fileWriter.write(cf[i] + "\tFinal Statistics:\n");
				fileWriter.write(maxFc1.getClassifier().toString()+
						maxFc2.getClassifier().toString()+
						maxFc3.getClassifier().toString()+
						maxFc4.getClassifier().toString());
				fileWriter.write("Array1's size: " +  finalArray1Size + "\t" + "accuracy: " + finalAccuracy1 +"\n");
				fileWriter.write("Array2's size: " +  finalArray2Size + "\t" + "accuracy: " + finalAccuracy2 +"\n");
				fileWriter.write("Array3's size: " +  finalArray3Size + "\t" + "accuracy: " + finalAccuracy3 +"\n");
				fileWriter.write("Array4's size: " +  finalArray4Size + "\t" + "accuracy: " + finalAccuracy4 +"\n");
				fileWriter.write("Max Correct Percentage: " +  maxCorrectPercentage);
				fileWriter.close(); 
	        }
            catch (IOException e)  
    	    {  
    	      e.printStackTrace();  
    	    }
        }
	}

	public static void converge(String cf) throws Exception {
		double Array1Accuracy = 0;
		double Array2Accuracy = 0;
		double Array3Accuracy = 0;
		double Array4Accuracy = 0;
		double accuracy1 = 0;
        double accuracy2 = 0;
        double accuracy3 = 0;
        double accuracy4 = 0;
		double lastAccuracy1;
		double lastAccuracy2;
		double lastAccuracy3;
		double lastAccuracy4;
		
        //System.out.println(instanceArray.size());
        //int shuffleTimes = 0;
		do{
			//System.out.println("Shuffling array 1 and 2 Time: " + shuffleTimes);
			Collections.shuffle(instanceArray);
			data1 = new Instances(allusers,0);
			data2 = new Instances(allusers,0);
			data3 = new Instances(allusers,0);
			data4 = new Instances(allusers,0);
			for (int i = 0; i < Math.round(instanceArray.size()/4); i++) {
				data1 = merge(data1, instanceArray.get(i));
			}
			for (int j = Math.round(instanceArray.size()/4); j < Math.round(instanceArray.size()/2); j++) {
				data2 = merge(data2, instanceArray.get(j));
			}
			for (int k = Math.round(instanceArray.size()/2); k < instanceArray.size()*3/4; k++) {
				data3 = merge(data3, instanceArray.get(k));
			}
			for (int k = Math.round(instanceArray.size()*3/4); k < instanceArray.size(); k++) {
				data4 = merge(data4, instanceArray.get(k));
			}
			//System.out.println(data1.size()/12);
			//System.out.println(data2.size()/12);
			fc1 = train(data1, cf);
			fc2 = train(data2, cf);
			fc3 = train(data3, cf);
			fc4 = train(data4, cf);
			//shuffleTimes++;
		}while ( fc1.numElements() == fc2.numElements() || fc1.numElements() == fc3.numElements() || fc2.numElements() == fc3.numElements());

		for (int expTimes = 0; expTimes < 200; expTimes++){
			Array1Accuracy = eval(fc1, data1, data1);
			Array2Accuracy = eval(fc2, data2, data2);
			Array3Accuracy = eval(fc3, data3, data3);
			Array4Accuracy = eval(fc4, data4, data4);
//			System.out.println("=============================================================================");
			System.out.println("Iteration: " + expTimes + "     maxCorrectPercentage: " + maxCorrectPercentage);
//			System.out.println("fc1:\n " + fc1.getClassifier().toString());
//			System.out.println("fc2:\n " + fc2.getClassifier().toString());
			System.out.println("fc1 size: " + fc1.numElements() + "\t" +
					"Array1's size: " + data1.numInstances()/12 + "\t" +
					"Array1's accuracy: " + Array1Accuracy);
			System.out.println("fc2 size: " + fc2.numElements() + "\t" +
					"Array2's size: " + data2.numInstances()/12 + "\t" +
					"Array2's accuracy: " + Array2Accuracy);
			System.out.println("fc3 size: " + fc3.numElements() + "\t" +
					"Array3's size: " + data3.numInstances()/12 + "\t" +
					"Array3's accuracy: " + Array3Accuracy);
			System.out.println("fc4 size: " + fc4.numElements() + "\t" +
					"Array4's size: " + data4.numInstances()/12 + "\t" +
					"Array4's accuracy: " + Array4Accuracy);
			ArrayList<Integer> data1Backup = new ArrayList<>();
			ArrayList<Integer> data2Backup = new ArrayList<>();
			ArrayList<Integer> data3Backup = new ArrayList<>();
			ArrayList<Integer> data4Backup = new ArrayList<>();
			for (int i = 0; i < data1.numInstances(); i = i + 12) {
				data1Backup.add((int)data1.instance(i).value(0));
			}
			for (int i = 0; i < data2.numInstances(); i = i + 12) {
				data2Backup.add((int)data2.instance(i).value(0));
			}
			for (int i = 0; i < data3.numInstances(); i = i + 12) {
				data3Backup.add((int)data3.instance(i).value(0));
			}
			for (int i = 0; i < data4.numInstances(); i = i + 12) {
				data4Backup.add((int)data4.instance(i).value(0));
			}
//			System.out.println("data1Backup: "+data1Backup.size());
//			System.out.println("data2Backup: "+data2Backup.size());
//			System.out.println("data3Backup: "+data3Backup.size());
//			System.out.println("data4Backup: "+data4Backup.size());
			Instances data1Temp = new Instances(data1);
			Instances data2Temp = new Instances(data2);
			Instances data3Temp = new Instances(data3);
			Instances data4Temp = new Instances(data4);
			
			lastAccuracy1 = Array1Accuracy;
			lastAccuracy2 = Array2Accuracy;
			lastAccuracy3 = Array3Accuracy;
			lastAccuracy4 = Array4Accuracy;
			
            data1.clear();
            data2.clear();
            data3.clear();
            data4.clear();
	
            for (int i = 0; i < allusers.numInstances(); i = i + 12) {
                Instances user = new Instances(allusers, i, 12);
                int userId = (int)user.instance(0).value(0);
//                if (data1Backup.contains(userId))
//        			System.out.println("In 1");
//                else if (data2Backup.contains(userId))
//        			System.out.println("In 2");
//        		else if (data3Backup.contains(userId))
//            		System.out.println("In 3");
//        		else if (data4Backup.contains(userId))
//            		System.out.println("In 4");	
                accuracy1 = eval(fc1, data1Temp, user);
                accuracy2 = eval(fc2, data2Temp, user);
                accuracy3 = eval(fc3, data3Temp, user);
                accuracy4 = eval(fc4, data4Temp, user);
               
                if (Math.max(accuracy1, accuracy2) > Math.max(accuracy3, accuracy4)){
                	if (accuracy1 > accuracy2)
                		data1 = merge(data1, user);
                	else if (accuracy1 < accuracy2)
                		data2 = merge(data2, user);
                	else if (accuracy1 == accuracy2)
    				{
                		//System.out.println(user.get(0));
                		if (data1Backup.contains(userId)) {
                			//System.out.println("In 1");
                			data1 = merge(data1, user);
                		}
                		else if (data2Backup.contains(userId)) {
                			//System.out.println("In 2");
                			data2 = merge(data2, user);
                		}
                		else {
	    					Random randomNum = new Random();
	    					if (randomNum.nextInt() % 2 == 0)
	    						data1 = merge(data1, user);
	    					else
	    						data2 = merge(data2, user);
                		}
    				}
                }
                else if (Math.max(accuracy1, accuracy2) < Math.max(accuracy3, accuracy4)){
                	if (accuracy3 > accuracy4)
                		data3 = merge(data3, user);
                	else if (accuracy3 < accuracy4)
                		data4 = merge(data4, user);
                	else if (accuracy3 == accuracy4)
    				{
                		if (data3Backup.contains(userId)) {
                			//System.out.println("In 3");
                			data3 = merge(data3, user);
                		}
                		else if (data4Backup.contains(userId)) {
                			//System.out.println("In 4");
                			data4 = merge(data4, user);
                		}
                		else {
	    					Random randomNum = new Random();
	    					if (randomNum.nextInt() % 2 == 0)
	    						data3 = merge(data3, user);
	    					else
	    						data4 = merge(data4, user);
                		}
    				}
                }
                else if (Math.max(accuracy1, accuracy4) > Math.max(accuracy2, accuracy3)){
                	if (accuracy1 > accuracy4)
                		data1 = merge(data1, user);
                	else if (accuracy1 < accuracy4)
                		data4 = merge(data4, user);
                	else if (accuracy1 == accuracy4)
    				{
                		if (data1Backup.contains(userId)) {
                			//System.out.println("In 1");
                			data1 = merge(data1, user);
                		}
                		else if (data4Backup.contains(userId)) {
                			//System.out.println("In 4");
                			data4 = merge(data4, user);
                		}
                		else {
	    					Random randomNum = new Random();
	    					if (randomNum.nextInt() % 2 == 0)
	    						data1 = merge(data1, user);
	    					else
	    						data4 = merge(data4, user);
                		}
    				}
                }
                else if (Math.max(accuracy1, accuracy4) < Math.max(accuracy2, accuracy3)){
                	if (accuracy2 > accuracy3)
                		data2 = merge(data2, user);
                	else if (accuracy2 < accuracy3)
                		data3 = merge(data3, user);
                	else if (accuracy2 == accuracy3)
    				{
                		if (data2Backup.contains(userId)) {
                			//System.out.println("In 2");
                			data2 = merge(data2, user);
                		}
                		else if (data3Backup.contains(userId)) {
                			//System.out.println("In 3");
                			data3 = merge(data3, user);
                		}
                		else {
	    					Random randomNum = new Random();
	    					if (randomNum.nextInt() % 2 == 0)
	    						data2 = merge(data2, user);
	    					else
	    						data3 = merge(data3, user);
                		}
    				}
                }
                else if (Math.max(accuracy1, accuracy3) >= Math.max(accuracy2, accuracy4)){
                	if (accuracy1 > accuracy3)
                		data1 = merge(data1, user);
                	else if (accuracy1 < accuracy3)
                		data3 = merge(data3, user);
                	else if (accuracy1 == accuracy3)
    				{
                		if (data1Backup.contains(userId)) {
                			//System.out.println("In 1");
                			data1 = merge(data1, user);
                		}
                		else if (data3Backup.contains(userId)) {
                			//System.out.println("In 3");
                			data3 = merge(data3, user);
                		}
                		else {
	    					Random randomNum = new Random();
	    					if (randomNum.nextInt() % 2 == 0)
	    						data1 = merge(data1, user);
	    					else
	    						data3 = merge(data3, user);
                		}
    				}
                }
                else if (Math.max(accuracy1, accuracy3) < Math.max(accuracy2, accuracy4)){
                	if (accuracy2 > accuracy4)
                		data2 = merge(data2, user);
                	else if (accuracy2 < accuracy4)
                		data4 = merge(data4, user);
                	else if (accuracy2 == accuracy4)
    				{
                		if (data2Backup.contains(userId)) {
                			//System.out.println("In 2");
                			data2 = merge(data2, user);
                		}
                		else if (data4Backup.contains(userId)) {
                			//System.out.println("In 4");
                			data4 = merge(data4, user);
                		}
                		else {
	    					Random randomNum = new Random();
	    					if (randomNum.nextInt() % 2 == 0)
	    						data2 = merge(data2, user);
	    					else
	    						data4 = merge(data4, user);
                		}
    				}
                }
            }
            
			if ( data1.size()==data1Temp.size() && 
					data2.size()==data2Temp.size() &&
					data3.size()==data3Temp.size() &&
					lastAccuracy1 == Array1Accuracy && 
					lastAccuracy2 == Array2Accuracy && 
					lastAccuracy3 == Array3Accuracy &&
					lastAccuracy4 == Array4Accuracy){
				double accuracy = (data1Backup.size()*Array1Accuracy + 
						data2Backup.size()*Array2Accuracy + 
						data3Backup.size()*Array3Accuracy +
						data4Backup.size()*Array4Accuracy)/1133;
				//System.out.println("*****************************************************************************");
				//System.out.println("Arrays converged within " + expTimes + " iterations");
				System.out.println("Cur Accuracy: " +  accuracy);
				//System.out.println("*****************************************************************************");
				if (accuracy > maxCorrectPercentage){
					maxCorrectPercentage = accuracy;
					maxFc1 = fc1;
					maxFc2 = fc2;
					maxFc3 = fc3;
					maxFc4 = fc4;
					finalArray1Size = data1.numInstances()/12;
					finalArray2Size = data2.numInstances()/12;
					finalArray3Size = data3.numInstances()/12;
					finalArray4Size = data4.numInstances()/12;
					finalAccuracy1 = Array1Accuracy;
					finalAccuracy2 = Array2Accuracy;
					finalAccuracy3 = Array3Accuracy;
					finalAccuracy4 = Array4Accuracy;
				}
				System.out.println("Max Accuracy: " +  maxCorrectPercentage);
				break;
			}

            fc1 = train(data1, cf);
            fc2 = train(data2, cf);
            fc3 = train(data3, cf);
            fc4 = train(data4, cf);
		}
	}

    public static FilteredClassifier train(Instances train, String cf) throws Exception
	{
        Remove rm = new Remove();
        String[] options = new String[2];
    	options[0] = "-C";
    	options[1] = cf;
    	
    	// REMOVING ID ATTRIBUTE AS THAT WON'T BE INPUT TO THE CLASSIFIER
        rm.setAttributeIndices("1"); 
        
        // classifier
        J48 j48 = new J48();
        j48.setOptions(options);
        // using an unpruned J48
        //j48.setUnpruned(true);        
        // meta-classifier
        
        FilteredClassifier cls = new FilteredClassifier();
        cls.setFilter(rm);
        cls.setClassifier(j48);
        train.setClassIndex((train.numAttributes()-1));
        cls.buildClassifier(train);
        return cls;
    }
    
	public static double eval(FilteredClassifier fc, Instances train, Instances test)  throws Exception
	{
		if (train.numInstances()==0) {
			return 0;
		}
		else {
			train.setClassIndex(train.numAttributes()-1);
			test.setClassIndex(test.numAttributes()-1);
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(fc, test);
			return eval.pctCorrect();
		}
	}

	
    public static Instances merge(Instances data1, Instances data2) throws Exception
    {
        // Check where are the string attributes
        int asize = data1.numAttributes();
        boolean strings_pos[] = new boolean[asize];
        for(int i=0; i<asize; i++)
        {
            Attribute att = data1.attribute(i);
            strings_pos[i] = ((att.type() == Attribute.STRING) ||
                              (att.type() == Attribute.NOMINAL));
        }

        // Create a new dataset
        Instances dest = new Instances(data1);
//            dest.setRelationName(data1.relationName() + "+" + data2.relationName());

        DataSource source = new DataSource(data2);
        Instances instances = source.getStructure();
        Instance instance = null;
        while (source.hasMoreElements(instances)) {
            instance = source.nextElement(instances);
            dest.add(instance);

            // Copy string attributes
            for(int i=0; i<asize; i++) {
                if(strings_pos[i]) {
                    dest.instance(dest.numInstances()-1)
                        .setValue(i,instance.stringValue(i));
                }
            }
        }
        return dest;
    }
}

