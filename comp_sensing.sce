clear;

//generating  measurements
n=10;
m=2;
x=full(sprand(n,1,0001,['normal'])); //generating sparse matrix
phi=rand(m,n,'normal'); //generating measurement matrix
y=phi*x; //generating compressed data

//global parameters
tau=2;  //no of neighbours
