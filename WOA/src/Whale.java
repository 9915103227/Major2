import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.WritableComparable;
public class Whale implements WritableComparable<Whale> 
{
	public Vector x[],y[];
	public double v[][];
	public double f,fitness;
	public Vector a[],r[];
	public double p;
	//public Whale best;
	public Whale()
	{
			super();
			fitness=Double.MAX_VALUE;
			//x=new Vector[];
	}
	public Whale(int l,int b)
	{
		x=new Vector[l];
		y=new Vector[l];
		a=new Vector[l];
		r=new Vector[l];
		v=new double[l][b];
		p=Math.random();
		for(int i=0;i<l;i++)
		{
			x[i]=new Vector(b);
			y[i]=new Vector(b);
			a[i]=new Vector(b);
			r[i]=new Vector(b);
			for(int j=0;j<b;j++)
			{
				a[i].vector[j]=2.00;
				r[i].vector[j]=Math.random();
			}
			
		}
	}
	public Whale(Whale Whale) {
		super();
		//this.Vector = new Vector(center.center);
		int l=Whale.x.length;
		int b=Whale.x[0].vector.length;
		this.p=Whale.p;
		this.x= new Vector[l];
		this.y= new Vector[l];
		this.r=new Vector[l];
		this.a=new Vector[l];
		for(int i=0;i<l;i++)
			{x[i]=new Vector(b); y[i]=new Vector(b);a[i]=new Vector(b);r[i]=new Vector(b);}
		v=new double[l][b];
		this.f=Whale.f;
		this.r=Whale.r;
		//this.A=Whale.A;
		this.fitness=Whale.fitness;
		for(int i=0;i<l;i++)
		{
			for(int j=0;j<b;j++)
			{
				this.x[i].vector[j]=Whale.x[i].vector[j];
				this.y[i].vector[j]=Whale.y[i].vector[j];
				this.r[i].vector[j]=Whale.r[i].vector[j];
				this.a[i].vector[j]=Whale.a[i].vector[j];
				this.v[i][j]=Whale.v[i][j];
			}
		}
		//this.best=Whale.best;
	}
//	public Whale(Vector center) {
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
		//out.writeDouble(f);
		//out.writeDouble(r);
		//out.writeDouble(A);
		out.writeDouble(fitness);
		
		for(int i=0;i<x.length;i++)
		{
			for(int j=0;j<x[0].vector.length;j++)
			{
				out.writeDouble(a[i].vector[j]);
			}
		}
		
		for(int i=0;i<x.length;i++)
		{
			for(int j=0;j<x[0].vector.length;j++)
			{
				out.writeDouble(r[i].vector[j]);
			}
		}
		
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
		a=new Vector[l];
		r=new Vector[l];
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
		//f=in.readDouble();
		//r=in.readDouble();
		//A=in.readDouble();
		fitness=in.readDouble();
		for(int i=0;i<l;i++)
		{
			a[i]=new Vector(b);
			for(int j=0;j<b;j++)
			a[i].vector[j]=in.readDouble();
		}
		
		for(int i=0;i<l;i++)
		{
			r[i]=new Vector(b);
			for(int j=0;j<b;j++)
			r[i].vector[j]=in.readDouble();
		}
		
	}
	@Override
	public int compareTo(Whale o) {
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
		
		if(this.f!=o.f||this.fitness!=o.fitness)
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
	return "Whale";
	}
}
