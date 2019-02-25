package com.clustering.mapreduce;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.LinkedList;
import java.util.List;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import com.clustering.model.Vector;
import com.clustering.model.Bat;
import com.clustering.model.DistanceMeasure;
public class BatMapper extends Mapper<LongWritable, Text, IntWritable, DoubleWritable>{
	List<Bat> centers = new LinkedList<Bat>();
	

@Override
protected void setup(Context context) throws IOException,
InterruptedException {
	//System.out.println("MAPPER WORKING");
super.setup(context);
Configuration conf = context.getConfiguration();
Path centroids = new Path("hdfs://localhost:9000/usr/seq_file/cen.seq");

FileSystem fs;
try {
	fs = FileSystem.get(new URI("hdfs://localhost:9000"),conf);

SequenceFile.Reader reader = new SequenceFile.Reader(fs, centroids, conf);


Bat key = new Bat() ;
IntWritable value = new IntWritable(-1);

while (reader.next(key,value)) {
	
centers.add(new Bat(key));
}
reader.close();

} catch (URISyntaxException e) {
	// TODO Auto-generated catch block
	e.printStackTrace();
}


}
@Override
protected void map(LongWritable key, Text val, Context context) throws IOException, InterruptedException {
//	System.out.println("Mapping");
	String x=val.toString();
	String tem[]=x.split(",");
	int j,k;
	
	Vector value=new Vector(tem.length);
	for(int i=0;i<tem.length;i++)
	{
	//	System.out.println(tem[i]);
		value.vector[i]=Double.parseDouble(tem[i]);
	}
	Configuration conf=context.getConfiguration();
	int iter=Integer.parseInt(conf.get("num.iteration"));
	int index=-1;
//	System.out.println("Next");
	for (Bat b : centers) {
		   index++;
            
         if(iter==1){
       
                double dist=DistanceMeasure.measureDistance(b.x[0], value);
              
                for(k=0;k<b.x.length;k++)
                {
                    double temp1=DistanceMeasure.measureDistance(b.x[k],value);
                    if(temp1<=dist)
                    {
                        dist=temp1;
                       
                    }
                  
                }
                DoubleWritable dd=new DoubleWritable(dist);
                
                IntWritable ind=new IntWritable(index);
                context.write(ind, dd);
            
           }
           else{
        	   double dist=DistanceMeasure.measureDistance(b.y[0], value);
               
               for(k=0;k<b.x.length;k++)
               {
                   double temp1=DistanceMeasure.measureDistance(b.y[k],value);
                  // System.out.println(temp1+" temp1");
                   if(temp1<=dist)
                   {
                       dist=temp1;
                      
                   }
                 
               }
               DoubleWritable dd=new DoubleWritable(dist);
               
               IntWritable ind=new IntWritable(index);
               context.write(ind, dd);
          
           }
	}
	
}
}
