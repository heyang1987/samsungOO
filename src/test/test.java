package test;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import samsung.arffFunctions;
import samsung.wekaFunctions;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class test {

	public static void main(String[] args) throws Exception {
        
        arffFunctions.generateArff(constantVar.cluster1_2, "docs/samsung_header.txt", "model1.arff");
        arffFunctions.generateArff(constantVar.cluster2_2, "docs/samsung_header.txt", "model2.arff");

        DataSource source1 = new DataSource("docs/model1.arff");
        DataSource source2 = new DataSource("docs/model2.arff");
        DataSource source3 = new DataSource("docs/model_multi.arff");

        Instances data1 = source1.getDataSet();
        Instances data2 = source2.getDataSet();
        Instances test = source3.getDataSet();
        
//        for (double cf = 0.25; cf > 0; cf = cf - 0.01) {
//        	FilteredClassifier cls = wekaFunctions.trainWithOption(data2, cf);
//        	String clsString = cls.getClassifier().toString();
//            //System.out.println(cls);
//            String str = clsString.substring(clsString.length()-4, clsString.length()-1).replaceAll(".*[^\\d](?=(\\d+))","");           
//            int treeSize = Integer.parseInt(str);
//        	double Array1Accuracy = wekaFunctions.evalCrossValidation(cls, data2);
//        	System.out.println(cf+"\t"+treeSize+"\t"+Array1Accuracy);
//        }
        FilteredClassifier cls1 = wekaFunctions.trainWithOption(data1, 0.01);
        FilteredClassifier cls2 = wekaFunctions.trainWithOption(data2, 0.01);
        //System.out.println(cls1);
        //System.out.println(cls2);
        
        Instances array1 = new Instances(test, 0);
        Instances array2 = new Instances(test, 0);
        Instances array3 = new Instances(test, 0);
        
        for (int i = 0; i < test.numInstances(); i = i + 12) {
            //int userID = (int)test.instance(i).value(0);
            Instances singleUserInstances = new Instances(test, i, 12);
            double acc1 = wekaFunctions.eval(cls1, data1, singleUserInstances);
            double acc2 = wekaFunctions.eval(cls2, data2, singleUserInstances);
            if (acc1> acc2) {
            	array1 = wekaFunctions.merge(array1, singleUserInstances);
            }
            else
            	array2 = wekaFunctions.merge(array2, singleUserInstances);
        }
        
        TextToFile("docs/model1new.arff",array1.toString());
        TextToFile("docs/model2new.arff",array2.toString());
        //TextToFile("docs/model3new.arff",array3.toString());
        for (double cf = 0.25; cf > 0; cf = cf - 0.01) {
			FilteredClassifier cls = wekaFunctions.trainWithOption(array1, cf);
			String clsString = cls.getClassifier().toString();
			//System.out.println(cls);
			String str = clsString.substring(clsString.length()-4, clsString.length()-1).replaceAll(".*[^\\d](?=(\\d+))","");           
			int treeSize = Integer.parseInt(str);
			double Array1Accuracy = wekaFunctions.evalCrossValidation(cls, array1);
			System.out.println(cf+"\t"+treeSize+"\t"+Array1Accuracy);
        }
        System.out.println();
        for (double cf = 0.25; cf > 0; cf = cf - 0.01) {
			FilteredClassifier cls = wekaFunctions.trainWithOption(array2, cf);
			String clsString = cls.getClassifier().toString();
			//System.out.println(cls);
			String str = clsString.substring(clsString.length()-4, clsString.length()-1).replaceAll(".*[^\\d](?=(\\d+))","");           
			int treeSize = Integer.parseInt(str);
			double Array1Accuracy = wekaFunctions.evalCrossValidation(cls, array2);
			System.out.println(cf+"\t"+treeSize+"\t"+Array1Accuracy);
        }
        

//        System.out.println("New Array1's Accuracy is: " + wekaFunctions.selfCVEval(array1) + " ("+ array1.size()/12 +")");
//        System.out.println("New Array2's Accuracy is: " + wekaFunctions.selfCVEval(array2) + " ("+ array2.size()/12 +")");
//        //System.out.println("New Array3's Accuracy is: " + wekaFunctions.selfCVEval(array3) + " ("+ array3.size()/12 +")");
//        System.out.println( (wekaFunctions.selfCVEval(array1)*array1.size()
//        		+wekaFunctions.selfCVEval(array2)*array2.size()
//        		)/(array1.size()+array2.size())
//        	);
        		
	}
	
	public static void TextToFile(final String strFilename, final String strBuffer) 
	{  
	    try  
	    {      
	    	// 创建文件对象  
	    	File fileText = new File(strFilename);  
	    	// 向文件写入对象写入信息  
	    	FileWriter fileWriter = new FileWriter(fileText);  
	  
	    	// 写文件        
	    	fileWriter.write(strBuffer);  
	    	// 关闭  
	    	fileWriter.close();  
	    }  
	    catch (IOException e)  
	    {  
	      e.printStackTrace();  
	    }
    }
}
