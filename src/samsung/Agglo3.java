package samsung;

import java.util.ArrayList;
import java.util.Collections;

import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Agglo3 {
    
    public static int classIndex = -1;
    public static int count_no;
    public static int count_yes;
    public static int count_multi;
    public static Instances noArrayInstances;
    public static Instances yesArrayInstances;
    public static Instances mulArrayInstances;
    public static ArrayList<Instances> multiInstanceArray = new ArrayList<>();
    public static String t1="N0 [label=\"0";
    public static String t2="N0 [label=\"1";

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

        DataSource source = new DataSource("docs/samsungNoSid.arff");
        Instances allusers=source.getDataSet();
        if (allusers.classIndex() == -1)
            classIndex=allusers.numAttributes()-1;
        allusers.setClassIndex(classIndex);
        //System.out.println(allusers.numInstances());
        noArrayInstances = new Instances(allusers, 0);
        yesArrayInstances = new Instances(allusers, 0);
        mulArrayInstances = new Instances(allusers, 0);

        for (int i = 0; i < allusers.numInstances(); i = i + 12) {
            //int userID = (int)allusers.instance(i).value(0);
            Instances singleUserInstances = new Instances(allusers, i, 12);
            
            J48 cls = trainWithOption(singleUserInstances, cf[24]);

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
                mulArrayInstances = wekaFunctions.merge(mulArrayInstances, singleUserInstances);
                //multiAccuracyList.add(wekaFunctions.eval(cls, singleUserInstances,singleUserInstances));               
            } 
        }
        System.out.println("'NO' node trees: " + count_no);
        System.out.println("'YES' node trees: " + count_yes);
        System.out.println("Multi-node trees: " + count_multi);
        
        double accuracyNo = Attitude.selfCVEval(noArrayInstances);
        double accuracyYes = Attitude.selfCVEval(yesArrayInstances);
        

        System.out.println("noArray's accuracy is: " + accuracyNo);
        System.out.println("yesArray's accuracy is: " + accuracyYes);
        System.out.println("mulArray's accuracy is: " + Attitude.selfCVEval(mulArrayInstances));
//        System.out.println(mulArrayInstances);
//        System.out.println(Attitude.trainWithOption(mulArrayInstances, cf[24]).toString());
//        
//        System.out.println(multiInstanceArray.size());
        System.out.println(mulArrayInstances.numInstances()/12);

        for (int i=0; i<25; i++) {
	        J48 fc = trainWithOption(mulArrayInstances, cf[i]);
	        String cls = fc.toString();
	        int treeSize = Integer.parseInt( cls.substring(cls.length()-4, cls.length()-1).replaceAll(".*[^\\d](?=(\\d+))","") );
	        //double accuracy = eval(fc, allusers, allusers);
	        double accuracyMult = Attitude.evalCrossValidation(fc, mulArrayInstances);
	        System.out.println(cf[i]+"\t"+treeSize+"\t"+(double)Math.round(accuracyMult*100)/100 );
	        System.out.println(fc);
	        System.out.println("Overall Accuracy over 3 profiles: "+ 
	        (accuracyNo*count_no+accuracyYes*count_yes+accuracyMult*count_multi)/1133);
	        System.out.println();
	        System.out.println();
		}
        
    }
    
    public static J48 trainWithOption(Instances train, String cf) throws Exception
	{
    	train.setClassIndex((train.numAttributes()-1));
    	
        String[] options = new String[2];
    	options[0] = "-C";
    	options[1] = cf;
        
    	//Init classifier
    	//Classifier cls = new J48();
    	J48 j48 = new J48();
        j48.setOptions(options);
    	j48.buildClassifier(train);
    	//cls.buildClassifier(train);
        return j48;
    }


}
