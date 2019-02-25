package com.clustering.mapreduce;

//import java.io.BufferedReader;
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
//import java.io.FileReader;
//import java.io.FileWriter;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Scanner;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Reader;
//import org.apache.hadoop.io.Text;
//import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
//import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import com.clustering.model.*;
//import org.apache.log4j.Logger;
//import org.apache.log4j.BasicConfigurator;
public class BatClustering {
	
private static final Log LOG = LogFactory.getLog(BatClustering.class);
static int pop=40; //population size
static int key_bat;
static Bat best_bat_t;
	public static void main(String[] args) throws IOException,InterruptedException, ClassNotFoundException, URISyntaxException {
		long startTime = System.currentTimeMillis();
		org.apache.log4j.BasicConfigurator.configure();
		System.setProperty("hadoop.home.dir", "/");
		int iteration = 1;
		Configuration conf = new Configuration();
		conf.set("num.iteration", iteration + "");   //no. of iterations
		double fmin=0,fmax=10.0;
		Path in = new Path("hdfs://localhost:9000/usr/input_file/data.txt"); 
		Path center = new Path("hdfs://localhost:9000/usr/seq_file/cen.seq");
		Path center1 = new Path("hdfs://localhost:9000/usr/seq_file/cen1.seq");
		Path out = new Path("hdfs://localhost:9000/usr/output/depth_1");
		Bat bat[]=new Bat[pop];
		int i,j,k;
		
         Path file1=new Path("hdfs://localhost:9000/usr/input_file/description.txt");
	FileSystem fs = FileSystem.get(new URI("hdfs://localhost:9000"),conf);
        BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(file1)));
        
        //Scanner sc=new Scanner(fs.open(file1));
        int d=Integer.parseInt(br.readLine());
        int p=Integer.parseInt(br.readLine());
        int c=Integer.parseInt(br.readLine());
        double min[]=new double[d];
        double max[]=new double[d];
        int ind=0;
        String ss="";
        while((ss=br.readLine())!=null)
        {
        	min[ind]=Double.parseDouble(ss);
        	ss=br.readLine();
        	max[ind++]=Double.parseDouble(ss);
        }
        i=0;
        
        
       
        //initializing
        double kmean[][]=new double[c][d];
    	
    	
    	
    	double best=Double.MAX_VALUE;
    	br=new BufferedReader(new InputStreamReader(fs.open(in)));
    	boolean C[][]=new boolean[c][p];
    	double data[][]=new double[p][d];
    	for(int mm=0;mm<p;mm++)
    	{String st[]=br.readLine().split(",");
        for(int m=0;m<d;m++)
        {
       	 data[mm][m]=Double.parseDouble(st[m]);
       
        }
        }
    	
    	
		
    for(int y=0;y<pop;y++)
    {
    	for(j=0;j<c;j++)
    	{
    		int rand=(int)(Math.random()*p);
    		for(k=0;k<d;k++)
    		{
    			kmean[j][k]=data[rand][k];  
        		}
    	}
    	
    	 bat[y]=new Bat(c,d);
          bat[y].fitness=0;
           bat[y].r=0.2;
           bat[y].A=0.9;
           for(int times=0;times<5;times++){
             for(i=0;i<p;i++)
             {
                 double min1=Double.MAX_VALUE;
                 for(j=0;j<c;j++)
                 {
                     double dist=distance(data[i],kmean[j]);
                  //   System.out.println(dist);
                     if(dist<min1)
                     {
                         min1=dist;
                         C[j][i]=true;
                         for(k=j-1;k>=0;k--)
                             C[k][i]=false;
                     }
                     else C[j][i]=false;
                 }
               
             }
             for(i=0;i<c;i++)
             {
                 int count=0;
                 double sum[]=new double[d];
                 for(j=0;j<p;j++)
                 {
                     if(C[i][j]==true)
                     { count++;
                         for(k=0;k<d;k++)
                         {
                             sum[k]=sum[k]+data[j][k];
                         }
                         
                     }
                 }
               
                 
                 for(k=0;k<d;k++)
                     {
                	if(count!=0) kmean[i][k]=sum[k]/count;
                    
                     	
                     }
             }
           }
             for(i=0;i<c;i++)
             {
            	 for(j=0;j<d;j++)
            	 {
            		 bat[y].x[i].vector[j]=kmean[i][j];
                  	bat[y].v[i][j]=0;
            	 }
            	 for(k=0;k<p;k++)
            	 {
            		 if(C[i][k])
            			 bat[y].fitness+=(distance(data[k],kmean[i]));
            	 }
            	
             }
             if(bat[y].fitness<=best)
             {
            	 key_bat=y;
            	 best=bat[y].fitness;
             }
         
         
    }
    
    Bat best_bat=new Bat(bat[key_bat]);
    best_bat.fitness=best;
     		
		
		if (fs.exists(out))
			fs.delete(out, true);
		
		if (fs.exists(in))
			fs.delete(out, true);
		
		
	
		int max_it=10;
		while (iteration <=max_it) {
			if(fs.exists(center))
				fs.delete(center,true);
			final SequenceFile.Writer centerWriter = SequenceFile.createWriter(fs,conf, center, Bat.class, IntWritable.class);
			final IntWritable value = new IntWritable(0);
	
			
			for(i=0;i<pop;i++)
            {    
				
                bat[i].f=fmin+ (fmax-fmin)*Math.random();
               if(iteration!=1){
                	
                
                for(j=0;j<c;j++)
                {       
                	
                    for(k=0;k<d;k++)
                    {
                        bat[i].v[j][k]=bat[i].v[j][k]+(bat[i].x[j].vector[k]-best_bat.x[j].vector[k])*Math.random();
                        bat[i].y[j].vector[k]=bat[i].x[j].vector[k]+bat[i].v[j][k];
                       
                        if(bat[i].y[j].vector[k]>max[k])
                        	bat[i].y[j].vector[k]=min[k]+Math.random()*(max[k]-min[k]); 
                        	
                        if(bat[i].y[j].vector[k]<min[k])
                        	bat[i].y[j].vector[k]=min[k]+Math.random()*(max[k]-min[k]);
                        	
                    }
                    
                }
          
           
                		
                	 if(Math.random()>=bat[i].r)
                    {
                       // double avg=Average(bat);
                        for(j=0;j<c;j++)
                        {
                            for(k=0;k<d;k++)
                            	
                            	//bat[i].y[j].vector[k]=best_bat.x[j].vector[k]+(-1+ Math.random()*2)*avg;
                            	bat[i].y[j].vector[k]=best_bat.x[j].vector[k]+(-1+ Math.random()*2)*0.001;
                        }
                        
                    }
            
                	}
            
            centerWriter.append(new Bat(bat[i]),value);
            }
			
			centerWriter.close();
	
			conf.set("counter", "0");	
			
			conf.set("num.iteration", iteration + "");
			Job job = new Job(conf);
			try{
			FileInputFormat.setInputPaths(job, in);
			FileOutputFormat.setOutputPath(job, out);
			job.setMapperClass(BatMapper.class);
			job.setReducerClass(BatReducer.class);
			
			job.setJarByClass(BatClustering.class);
			job.setMapOutputKeyClass(IntWritable.class);
			job.setMapOutputValueClass(DoubleWritable.class);
			job.setOutputKeyClass(Bat.class);
			job.setOutputValueClass(DoubleWritable.class);
			job.setInputFormatClass(TextInputFormat.class);
			job.setOutputFormatClass(SequenceFileOutputFormat.class);
			
			
			
			
			
			job.waitForCompletion(true);}
			catch(Exception e)
			{
				System.out.println("ERROR");
				e.printStackTrace(System.err);
			}
			double best_val=Merge(conf);
			
			SequenceFile.Reader reader=new Reader(fs,center1,conf);
			int indd=0;
			Bat ke=new Bat();
			DoubleWritable v=new DoubleWritable(0);
			while(reader.next(ke,v))
			{
				
				bat[indd++]=new Bat(ke);
				

			}
			reader.close();
			iteration++;
			
			
			if(best_val<best){ best=best_val;
			best_bat=new Bat(best_bat_t);}
			out = new Path("hdfs://localhost:9000/usr/output/depth_" +
					iteration);
			if(fs.exists(out))
				fs.delete(out, true);
			fmax= fmax-iteration*(fmax/max_it);
		}
		
		System.out.println(best);
		long stopTime = System.currentTimeMillis();
	      
	      System.out.println(stopTime - startTime);
	}
	public static double Merge(Configuration conf)throws IOException, URISyntaxException
	{
		double best=Double.MAX_VALUE;
		key_bat=-1;
		FileSystem fs = FileSystem.get(new URI("hdfs://localhost:9000"),conf);
	
		Path center1 = new Path("hdfs://localhost:9000/usr/seq_file/cen1.seq");

		SequenceFile.Reader reader = new SequenceFile.Reader(fs, center1, conf);

		Bat key = new Bat() ;
		DoubleWritable value = new DoubleWritable(-1);
		
		
		while (reader.next(key,value)) {
			
		if(best>value.get())
		{
			best=value.get();
			best_bat_t=new Bat(key);
		}
		
		}
		
		reader.close();
		
		
		return best;
	}
		public static double Average(Bat bat[])
		{
			
			double avg=0;
			for(int i=0;i<bat.length;i++)
				avg=avg+bat[i].A;
			avg=avg/bat.length;
			return avg;
		}
		public static double distance(double st[],double kmean[])
		{
			double dist=0;
			for(int i=0;i<st.length;i++)
			{
				dist+=Math.pow(st[i]-kmean[i],2);
			}
			return dist;
		}
}
