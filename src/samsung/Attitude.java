package samsung;

import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Attitude {

    public static int classIndex = -1;
    public static Random random = new Random();
	
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
		
        DataSource source = new DataSource("docs/6Cluster.arff");
        Instances allusers=source.getDataSet();
        if (allusers.classIndex() == -1)
            classIndex=allusers.numAttributes()-1;
        allusers.setClassIndex(classIndex);
        
        int i = 0;
        for (i=0; i<25; i++) {
	        Classifier fc = trainWithOption(allusers, cf[i]);
	        System.out.println(fc);
	        String cls = fc.toString();
	        int treeSize = Integer.parseInt( cls.substring(cls.length()-4, cls.length()-1).replaceAll(".*[^\\d](?=(\\d+))","") );
	        double accuracy = eval(fc, allusers, allusers);
	        System.out.println("&5"+"\t&"+cf[i]+"\t&"+treeSize+"\t&"+(double)Math.round(accuracy*100)/100+" \\\\ \\cline{2-5}" );
		}
	}
	
    public static Classifier trainWithOption(Instances train, String cf) throws Exception
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
    
	public static double eval(Classifier fc, Instances train, Instances test)  throws Exception
	{
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(fc, test);
		return eval.pctCorrect();
	}
    
	public static double evalCrossValidation(Classifier cls, Instances data) throws Exception
	{
		data.setClassIndex((data.numAttributes()-1));
		//Random random = new Random();
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(cls, data, 10, random);
		return eval.pctCorrect();
	}
	
    public static double selfCVEval(Instances data) throws Exception
	{
	    data.setClassIndex((data.numAttributes()-1));
	    //Random random = new Random();
	    Evaluation eval = new Evaluation(data);
	    eval.crossValidateModel(new J48(), data, 10, random);
	    return eval.pctCorrect();
	}

}
