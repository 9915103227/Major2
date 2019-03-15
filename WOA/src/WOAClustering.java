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
//import com.clustering.model.*;


//import org.apache.log4j.Logger;
//import org.apache.log4j.BasicConfigurator;
public class WOAClustering {
	
private static final Log LOG = LogFactory.getLog(WOAClustering.class);
static int pop=100; //population size
static int key_Whale;
static Whale best_Whale_t;
	public static void main(String[] args) throws IOException,InterruptedException, ClassNotFoundException, URISyntaxException {
		double reducing;
		long startTime = System.currentTimeMillis();
		//org.apache.log4j.BasicConfigurator.configure();
		System.setProperty("hadoop.home.dir", "/");
		int iteration = 1;
		Configuration conf = new Configuration();
		conf.set("num.iteration", iteration + "");   //no. of iterations
		double fmin=0,fmax=10.0;
		int users=1000;
		int movies=1700;
		Path in = new Path("hdfs://localhost:9000/usr/input_file/data.txt"); 
		Path center = new Path("hdfs://localhost:9000/usr/seq_file/cen.seq");
		Path center1 = new Path("hdfs://localhost:9000/usr/seq_file/cen1.seq");
		Path out = new Path("hdfs://localhost:9000/usr/output/depth_1");
		Whale Whale[]=new Whale[pop];
		int i,j,k;
		
         Path file1=new Path("hdfs://localhost:9000/usr/input_file/description.txt");
	FileSystem fs = FileSystem.get(new URI("hdfs://localhost:9000/"),conf);
        BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(file1)));
        
        //Scanner sc=new Scanner(fs.open(file1));
        int d=Integer.parseInt(br.readLine());//number of columns
        int p=Integer.parseInt(br.readLine());//number of data points
        int c=Integer.parseInt(br.readLine());//might be number of centroids
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
    	{	
    		String st[]=br.readLine().split(",");
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
	    	
	    	 Whale[y]=new Whale(c,d);
	         Whale[y].fitness=0;
	         //Whale[y].r=0.2;
	         //Whale[y].A=0.9;
	         for(int times=0;times<5;times++)
	         {
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
	          		 Whale[y].x[i].vector[j]=kmean[i][j];
	                 Whale[y].v[i][j]=0;
	          	 }
	          	 for(k=0;k<p;k++)
	          	 {
	          		 if(C[i][k])
	            	 Whale[y].fitness+=(distance(data[k],kmean[i]));
	          	 }
	            	
	           }
	           if(Whale[y].fitness<=best)
	           {
	           	 key_Whale=y;
	           	 best=Whale[y].fitness;
	           }         
	    }
    
	    Whale best_Whale=new Whale(Whale[key_Whale]);
	    best_Whale.fitness=best;
	     		
			
		if (fs.exists(out))
			fs.delete(out, true);
			
		if (fs.exists(in))
			fs.delete(out, true);
			
			
		
		int max_it=10;
		reducing=2/max_it;
		while (iteration <=max_it)
		{
			if(fs.exists(center))
				fs.delete(center,true);
			final SequenceFile.Writer centerWriter = SequenceFile.createWriter(fs,conf, center, Whale.class, IntWritable.class);
			final IntWritable value = new IntWritable(0);
		
				
			for(i=0;i<pop;i++)
	        {    			
	          Whale[i].f=fmin+ (fmax-fmin)*Math.random();
	          Vector D[] =new Vector[c];
	          Vector Ceq[] =new Vector[c];
	          Vector A[] =new Vector[c];
	          if(iteration!=1)
	          {               
	           for(j=0;j<c;j++)
	           { 
	        	   D[j]=new Vector(d);
	        	   Ceq[j]=new Vector(d);
	        	   A[j]=new Vector(d);
	             for(k=0;k<d;k++)
	             {
	            	 if(Whale[i].p>=0.5)
	            	 {//EQUATION 2.5
	            		D[j].vector[k]=Math.abs(Whale[i].x[j].vector[k]-best_Whale.x[j].vector[k]);
	            		double l=Math.random();
	            		Whale[i].y[j].vector[k]=D[j].vector[k]*Math.exp(0.5*l)*Math.cos(2*3.14*l)+best_Whale.x[j].vector[k];
	            	 }
	            	 else
	            	 {
	            		
	            		 double aA=Whale[i].a[j].vector[k];
	            		 double rA=Whale[i].r[j].vector[k];
	            		 A[j].vector[k]=2*aA*rA-aA;
	            		 if(Math.abs(A[j].vector[k])<1)
	            		 {
	            			 //EQUATION 2.1
	            			 Ceq[j].vector[k]=2*Whale[i].r[j].vector[k];
	            			 D[j].vector[k]=Math.abs(Ceq[j].vector[k]*best_Whale.x[j].vector[k]-Whale[i].x[j].vector[k]);
	            			 Whale[i].y[j].vector[k]=best_Whale.x[j].vector[k]-A[j].vector[k]*D[j].vector[k];
	            		 }
	            		 else
	            		 {
	            			 //EQUATION 2.8
	            			 Ceq[j].vector[k]=2*Whale[i].r[j].vector[k];
	            			 double xRand=Math.random();
	            			 D[j].vector[k]=Math.abs(Ceq[j].vector[k]*xRand-Whale[i].x[j].vector[k]);
	            			 Whale[i].y[j].vector[k]=xRand-A[j].vector[k]*D[j].vector[k];
	            		 }
	            	 }
	            	 
	               Whale[i].v[j][k]=Whale[i].v[j][k]+(Whale[i].x[j].vector[k]-best_Whale.x[j].vector[k])*Math.random();
	               Whale[i].y[j].vector[k]=Whale[i].x[j].vector[k]+Whale[i].v[j][k];
	               if(Whale[i].y[j].vector[k]>max[k])
	               	{Whale[i].y[j].vector[k]=min[k]+Math.random()*(max[k]-min[k]); }
	               if(Whale[i].y[j].vector[k]<min[k])
	               {	Whale[i].y[j].vector[k]=min[k]+Math.random()*(max[k]-min[k]);}
	             }
	            }
	          
	           
	                		
	            /*if(Math.random()>=Whale[i].r)
	            {
	                     // double avg=Average(Whale);
	              for(j=0;j<c;j++)
	              {
	                for(k=0;k<d;k++)
	                            	
	                //Whale[i].y[j].vector[k]=best_Whale.x[j].vector[k]+(-1+ Math.random()*2)*avg;
	                Whale[i].y[j].vector[k]=best_Whale.x[j].vector[k]+(-1+ Math.random()*2)*0.001;
	               }
	            }*/
	           }
	            
	           centerWriter.append(new Whale(Whale[i]),value);
	        }
				
			centerWriter.close();
			conf.set("counter", "0");	
			conf.set("num.iteration", iteration + "");
			Job job = new Job(conf);
			try{
				FileInputFormat.setInputPaths(job, in);
				FileOutputFormat.setOutputPath(job, out);
				job.setMapperClass(WhaleMapper.class);
				job.setReducerClass(WhaleReducer.class);
				
				job.setJarByClass(WOAClustering.class);
				job.setMapOutputKeyClass(IntWritable.class);
				job.setMapOutputValueClass(DoubleWritable.class);
				job.setOutputKeyClass(Whale.class);
				job.setOutputValueClass(DoubleWritable.class);
				job.setInputFormatClass(TextInputFormat.class);
				job.setOutputFormatClass(SequenceFileOutputFormat.class);
				
				
				
				
				
				job.waitForCompletion(true);
			}
			catch(Exception e)
			{
				System.out.println("ERROR");
				e.printStackTrace(System.err);
			}
			double best_val=Merge(conf);
			
			SequenceFile.Reader reader=new Reader(fs,center1,conf);
			int indd=0;
			Whale ke=new Whale();
			DoubleWritable v=new DoubleWritable(0);
			while(reader.next(ke,v))
			{	
				Whale[indd++]=new Whale(ke);
			}
			reader.close();
			iteration++;
				
				
			if(best_val<best)
			{
				best=best_val;
				best_Whale=new Whale(best_Whale_t);
			}
			out = new Path("hdfs://localhost:9000/usr/output/depth_" + iteration);
			if(fs.exists(out))
				fs.delete(out, true);
			fmax= fmax-iteration*(fmax/max_it);
			for(int jIter=0;jIter<pop;jIter++)
			{
				for(int alpha=0;alpha<c;alpha++)
				{
					for(int beta=0;beta<d;beta++)
					{
						Whale[jIter].a[alpha].vector[beta]-=reducing;
					}
				}
			}
		}
		
		System.out.println(best);
		long stopTime = System.currentTimeMillis();
		double mae=0.00;
	    for(int user=0;i<users;i++)
	    {
	    	double minDist=Double.MAX_VALUE;
	    	int bestCentroid=0;
	    	for(int centroid=0;centroid<c;centroid++)
	    	{
	    		double distDataCentr=distance(data[user], kmean[centroid]);
	    		if(distDataCentr<minDist)
	    		{
	    			minDist=distDataCentr;
	    			bestCentroid=centroid;
	    		}
	    	}
	    	for(int movie=0;movie<movies;movie++)
	    	{
	    		mae=mae+Math.abs(kmean[bestCentroid][movie]-data[user][movie]);
	    	}
	    }
	    mae/=(users*movies);
	    System.out.println(mae);
	    System.out.println(stopTime - startTime);
	    
	}
	public static double Merge(Configuration conf)throws IOException, URISyntaxException
	{
		double best=Double.MAX_VALUE;
		key_Whale=-1;
		FileSystem fs = FileSystem.get(new URI("hdfs://localhost:9000"),conf);
	
		Path center1 = new Path("hdfs://localhost:9000/usr/seq_file/cen1.seq");

		SequenceFile.Reader reader = new SequenceFile.Reader(fs, center1, conf);

		Whale key = new Whale() ;
		DoubleWritable value = new DoubleWritable(-1);
		
		
		while (reader.next(key,value)) {
			
			if(best>value.get())
			{
				best=value.get();
				best_Whale_t=new Whale(key);
			}
		
		}
		
		reader.close();
		
		
		return best;
	}
		/*public static double Average(Whale Whale[])
		{
			
			double avg=0;
			for(int i=0;i<Whale.length;i++)
				avg=avg+Whale[i].A;
			avg=avg/Whale.length;
			return avg;
		}*/
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
