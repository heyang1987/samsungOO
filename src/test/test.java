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
        
        arffFunctions.generateArff(constantVar.cluster436_10_1, "docs/samsung_header.txt", "model436_10_1.arff");
        arffFunctions.generateArff(constantVar.cluster436_10_2, "docs/samsung_header.txt", "model436_10_2.arff");

        DataSource source1 = new DataSource("docs/model436_10_1.arff");
        DataSource source2 = new DataSource("docs/model436_10_2.arff");
        DataSource source3 = new DataSource("docs/model_multi.arff");

        Instances data1 = source1.getDataSet();
        Instances data2 = source2.getDataSet();
        Instances test = source3.getDataSet();
        
        FilteredClassifier cls1 = wekaFunctions.trainWithOption(data1, 0.25);
        FilteredClassifier cls2 = wekaFunctions.trainWithOption(data2, 0.25);
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
        
        TextToFile("docs/model436_10_1new.arff",array1.toString());
        TextToFile("docs/model436_10_2new.arff",array2.toString());
        //TextToFile("docs/model3new.arff",array3.toString());
        
        java.text.DecimalFormat   df   =new   java.text.DecimalFormat("0.00");
        for (double cf = 0.25; cf > 0; cf = cf - 0.01) {
			FilteredClassifier cls1new = wekaFunctions.trainWithOption(array1, cf);
			String clsString1 = cls1new.getClassifier().toString();
			//System.out.println(cls);
			String str1 = clsString1.substring(clsString1.length()-4, clsString1.length()-1).replaceAll(".*[^\\d](?=(\\d+))","");           
			int treeSize1 = Integer.parseInt(str1);
			double Array1Accuracy1 = wekaFunctions.evalCrossValidation(cls1new, array1);


			FilteredClassifier cls2new = wekaFunctions.trainWithOption(array2, cf);
			String clsString2 = cls2new.getClassifier().toString();
			//System.out.println(cls);
			String str2 = clsString2.substring(clsString2.length()-4, clsString2.length()-1).replaceAll(".*[^\\d](?=(\\d+))","");           
			int treeSize2 = Integer.parseInt(str2);
			double Array1Accuracy2 = wekaFunctions.evalCrossValidation(cls2new, array2);
			System.out.println(df.format(cf)+"\t"+treeSize1+"\t"+Array1Accuracy1+"\t"+treeSize2+"\t"+Array1Accuracy2+"\t"+(Array1Accuracy1*array1.numInstances()+Array1Accuracy2*array2.numInstances())/(array1.numInstances()+array2.numInstances()));
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
