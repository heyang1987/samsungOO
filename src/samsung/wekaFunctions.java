package samsung;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;

public class wekaFunctions {
	
    public static FilteredClassifier getFilteredClassifier() throws Exception
    {
        Remove rm = new Remove();  	
        rm.setAttributeIndices("1");  // REMOVING ID ATTRIBUTE AS THAT WON'T BE INPUT TO THE CLASSIFIER
        String[] options = new String[2];
    	options[0] = "-C";
    	options[1] = "0.25";
        // classifier
        J48 j48 = new J48();
        j48.setOptions(options);
        //j48.setUnpruned(true);        // using an unpruned J48
        // meta-classifier
        FilteredClassifier cls = new FilteredClassifier();
        cls.setFilter(rm);
        cls.setClassifier(j48);
        return cls;
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
        
    public static FilteredClassifier train(Instances train) throws Exception
	{
        FilteredClassifier fc = getFilteredClassifier();
        train.setClassIndex((train.numAttributes()-1));
        fc.buildClassifier(train);
        return fc;
    }
    
    public static FilteredClassifier trainWithOption(Instances train, double cf) throws Exception
	{
        Remove rm = new Remove();
        String[] options = new String[2];
    	options[0] = "-C";
    	options[1] = Double.toString(cf);
    	
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

    public static FilteredClassifier trainWithOption(Instances train, String cf) throws Exception
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
        
    public static double trainSelfEval(Instances data) throws Exception
	{
            if (data.classIndex() == -1)
                data.setClassIndex((data.numAttributes()-1));

            FilteredClassifier cls = getFilteredClassifier();
            cls.buildClassifier(data);
            // evaluation
            Evaluation eval = new Evaluation(data);
            eval.evaluateModel(cls, data);
            return eval.pctCorrect();
	}
        
    public static double selfCVEval(Instances data) throws Exception
	{
            data.setClassIndex((data.numAttributes()-1));

            FilteredClassifier cls = getFilteredClassifier();
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(cls, data, 10, new Random(1));
            return eval.pctCorrect();
	}
	
	public static double eval(FilteredClassifier fc, Instances train, Instances test)  throws Exception
	{
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(fc, test);
		return eval.pctCorrect();
	}

	public static double evalCrossValidation(FilteredClassifier fc, Instances data) throws Exception
	{
		Random random = new Random();
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(fc, data, 10, random);
		return eval.pctCorrect();
	}
        
	public static double trainAndEval(Instances train, Instances test, int classIndex) throws Exception{
            train.setClassIndex(classIndex);		
            FilteredClassifier fc = getFilteredClassifier();
            // train
            fc.buildClassifier(train);
            // evaluation
            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(fc, test);
            return eval.pctCorrect();	
	}
        
        public static double trainAndEvalNoPrune(Instances train, Instances test, int classIndex) throws Exception{
            train.setClassIndex(classIndex);		
            FilteredClassifier fc = getFilteredClassifier();
            // train
            fc.buildClassifier(train);
            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(fc, test);
            return eval.pctCorrect();	
	}
        
        public static Instances trim(Instances data, int classIndex){
                //int count = 0;
                for (int i = data.numInstances()-1; i>=0; i--){
                        //System.out.println(data.instance(i).stringValue(classIndex-1));
                        if (!data.instance(i).stringValue(classIndex-1).equals("always")){
                                //count++;
                                data.delete(i);
                        }
                }
                //System.out.println("Not ALWALS INSTANCES #: "+count);
		return data;
    }
}
