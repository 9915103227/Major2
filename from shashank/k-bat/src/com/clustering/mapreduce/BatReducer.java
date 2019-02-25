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
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Mapper.Context;

import com.clustering.model.*;

public class BatReducer extends Reducer<IntWritable, DoubleWritable, Bat, DoubleWritable>{
	List<Bat> centers = new LinkedList<Bat>();
	double alpha=0.9, gama=0.01;
	SequenceFile.Writer centerWriter;
	protected void setup(Context context) throws IOException, InterruptedException {	
		super.setup(context);
		Configuration conf = context.getConfiguration();
		Path centroids = new Path("hdfs://localhost:9000/usr/seq_file/cen.seq");
		try{
			FileSystem fs = FileSystem.get(new URI("hdfs://localhost:9000"),conf);
		
		
		SequenceFile.Reader reader = new SequenceFile.Reader(fs, centroids, conf);
	
	
		Bat key = new Bat() ;
		IntWritable value = new IntWritable(-1);
		
		while (reader.next(key,value)) {
		
		centers.add(new Bat(key));
		}
		reader.close();
		Path center1=new Path("hdfs://localhost:9000/usr/seq_file/cen1.seq");
		if(fs.exists(center1))
				fs.delete(center1, true);
			//final SequenceFile.Writer centerWriter1 = SequenceFile.createWriter(fs,conf, center1, Bat.class, IntWritable.class);
		centerWriter = SequenceFile.createWriter(fs,conf, center1, Bat.class, DoubleWritable.class);
		}
		catch(URISyntaxException e){}

		

	}
	protected void reduce(IntWritable ke, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException{
		
		int index=0;
		Bat key=new Bat();
		for(Bat b:centers)
		{
			if(index==ke.get())
				{key=new Bat(b);
				break;
				}
			index++;
		}
		
	
		double sum=0;
			for(DoubleWritable value :values){
			
				sum=sum+value.get();
			}
			
		
			Configuration conf =context.getConfiguration();
			int t=Integer.parseInt(conf.get("num.iteration"));
			
			if(t==1) key.fitness=sum;
			else{
				
			if(sum<=key.fitness)
			{
				key.fitness=sum;
				for(int i=0;i<key.x.length;i++)
					for(int j=0;j<key.x[0].vector.length;j++)
						key.x[i].vector[j]=key.y[i].vector[j];
				key.A=key.A*alpha;
				key.r=0.5*(1-Math.exp(-gama*t));
				
			
				
			}}
		try {
				
			

			
			centerWriter.append(new Bat(key),new DoubleWritable(key.fitness));
			
			int ti=Integer.parseInt(conf.get("counter"));
		
				ti++;
				conf.set("counter", ti + "");
				context.write(key, new DoubleWritable(key.fitness));
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			
			
	}
	protected void cleanup(Context context) throws IOException,InterruptedException{
		
		super.cleanup(context);
		centerWriter.close();
	//	System.out.println("clean");
	}
}
