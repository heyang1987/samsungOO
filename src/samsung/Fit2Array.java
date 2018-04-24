package samsung;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Fit2Array {
	
	public static ArrayList<Instances> instanceArray = new ArrayList<>();
	public static ArrayList<Instances> array1 = new ArrayList<>();
	public static ArrayList<Instances> array2 = new ArrayList<>();
	public static Instances array1Instances;
	public static Instances array2Instances;

	public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("docs/samsung.arff");
        Instances allusers=source.getDataSet();
        if (allusers.classIndex() == -1)
            allusers.setClassIndex(allusers.numAttributes()-1);
        //System.out.println(allusers.numInstances());
        array1Instances = new Instances(allusers, 0);
        array2Instances = new Instances(allusers, 0);
        for (int i = 0; i < allusers.numInstances(); i = i + 12) {
            //int userID = (int)allusers.instance(i).value(0);
            Instances singleUserInstances = new Instances(allusers, i, 12);
            instanceArray.add(singleUserInstances);
        }
        System.out.println(instanceArray.size());
        int shuffleTimes = 0;
		do{
			System.out.println("Shuffling array 1 and 2 Time: " + shuffleTimes);
			Collections.shuffle(instanceArray);
			array1 = new ArrayList<>(instanceArray.subList(0, Math.round(instanceArray.size()/2)));
			array2 = new ArrayList<>(instanceArray.subList(Math.round(instanceArray.size()/2), instanceArray.size()));
			
			//System.out.println(array1);
			//System.out.println(array2);
			
			
			for (int i = 0; i < array1.size(); i++) {
				array1Instances = wekaFunctions.merge(array1Instances, array1.get(i));
			}
			for (int j = 1; j < array2.size(); j++) {
				array2Instances = wekaFunctions.merge(array2Instances, array2.get(j));
			}
			System.out.println(array1Instances.size()/12);
			System.out.println(array2Instances);
		
			FilteredClassifier fc1 = wekaFunctions.train(array1Instances);
			FilteredClassifier fc2 = wekaFunctions.train(array2Instances);
                        
			//System.out.println("fc1 size: " + fc1.numElements());
			//System.out.println("fc2 size: " + fc2.numElements());

			shuffleTimes++;
		}while ( (fc1.numElements() == 1) && (fc2.numElements() == 1) );
	}

}
