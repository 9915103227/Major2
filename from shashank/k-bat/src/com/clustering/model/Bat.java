package com.clustering.model;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.WritableComparable;
public class Bat implements WritableComparable<Bat> 
{
	public Vector x[],y[];
	public double v[][];
	public double f,r,A,fitness;
	//public Bat best;
	public Bat()
	{
			super();
			fitness=Double.MAX_VALUE;
			//x=new Vector[];
	}
	public Bat(int l,int b)
	{
		x=new Vector[l];
		y=new Vector[l];
		v=new double[l][b];
		for(int i=0;i<l;i++)
		{
			x[i]=new Vector(b);
			y[i]=new Vector(b);
		}
	}
	public Bat(Bat bat) {
		super();
		//this.Vector = new Vector(center.center);
		int l=bat.x.length;
		int b=bat.x[0].vector.length;
		this.x= new Vector[l];
		this.y= new Vector[l];
		for(int i=0;i<l;i++)
			{x[i]=new Vector(b); y[i]=new Vector(b);}
		v=new double[l][b];
		this.f=bat.f;
		this.r=bat.r;
		this.A=bat.A;
		this.fitness=bat.fitness;
		for(int i=0;i<l;i++)
		{
			for(int j=0;j<b;j++)
			{
				this.x[i].vector[j]=bat.x[i].vector[j];
				this.y[i].vector[j]=bat.y[i].vector[j];
				this.v[i][j]=bat.v[i][j];
			}
		}
		//this.best=bat.best;
	}
//	public Bat(Vector center) {
	//	super();
		//this.x = center;
//	}
	
	@Override
	public void write(DataOutput out) throws IOException {
		int l=x.length;
		int b=v[0].length;
		out.writeInt(l);
		out.writeInt(b);
		for(int i=0;i<l;i++)
		{
			for(int j=0;j<b;j++)
				out.writeDouble(x[i].vector[j]);
		}
		for(int i=0;i<l;i++)
		{
			for(int j=0;j<b;j++)
				out.writeDouble(y[i].vector[j]);
			
		}
		for(int i=0;i<x.length;i++)
		{
			for(int j=0;j<x[0].vector.length;j++)
			{
				out.writeDouble(v[i][j]);
			}
		}
		out.writeDouble(f);
		out.writeDouble(r);
		out.writeDouble(A);
		out.writeDouble(fitness);
		
	}
	@Override
	public void readFields(DataInput in) throws IOException {
		//this.center = new Vector();
		//center.readFields(in);
		int l = in.readInt();
		int b= in.readInt();
		x=new Vector[l];
		y=new Vector[l];
		v=new double[l][b];
		for(int i=0;i<l;i++)
		{
			x[i]=new Vector(b);
			for(int j=0;j<b;j++)
			x[i].vector[j]=in.readDouble();
		}
		for(int i=0;i<l;i++)
		{
			y[i]=new Vector(b);
			for(int j=0;j<b;j++)
				y[i].vector[j]=in.readDouble();
		}
		for(int i=0;i<l;i++)
		{
			for(int j=0;j<b;j++)
			v[i][j]=in.readDouble();
		}
		f=in.readDouble();
		r=in.readDouble();
		A=in.readDouble();
		fitness=in.readDouble();
		
	}
	@Override
	public int compareTo(Bat o) {
		int l=x.length;
		int b=x[0].vector.length;
		int i,j;
		for(i=0;i<l;i++)
		{
			if(this.x[i].compareTo(o.x[i])<0)
				return -1;
			if(this.x[i].compareTo(o.x[i])>0)
				return 1;
		}
		for(i=0;i<l;i++)
		{
			for(j=0;j<b;j++)
			{
			if(this.v[i][j]<o.v[i][j])
				return -1;
			if(this.v[i][j]>o.v[i][j])
				return 1;
			}
		}
		
		if(this.r!=o.r||this.A!=o.A||this.f!=o.f||this.fitness!=o.fitness)
			return 1;
		return 0;
		
	}
/**
* @return the center
*/
	public Vector[] getCenter() {
		return x;
	}
@Override
	public String toString() {
		//return "ClusterCenter [center=" + center + "]";
	return "bat";
	}
}
