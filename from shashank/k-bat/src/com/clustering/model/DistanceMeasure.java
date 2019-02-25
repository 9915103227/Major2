package com.clustering.model;

public class DistanceMeasure {
public static final double measureDistance(Vector center, Vector v) 
{
double sum = 0;
int length = v.getVector().length;
for (int i = 0; i < length; i++) {
sum += Math.pow((center.getVector()[i] - v.getVector()[i]),2);
}
return Math.sqrt(sum);
}
}
