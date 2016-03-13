//Implmenting algorithm in Huang-Learning with dynamic group sparsity
//1D case

clear;
//Defining constants
n=512; //size of data
k=64; //sparsity
q=4; //no of clusters
m=3*k; //no of measurements
sigma=0.01; //measurement noise
tau=2; //no of neighbours
k_min=1;
k_max=n/3; //range of k
steps=50; //steps to increase k
del_k=(k_max-k_min)/steps; //step size for increasing k
//Generating phi
phi=rand(m,n,'normal');

//normalizing rows of phi
max_phi=max(abs(phi),'c');
for i=1:m
    if abs(max_phi(i))>1 then
        for j=1:n
        phi(i,j)=phi(i,j)/abs(max_phi(i));
        end
    end
end

//Generating measurements
x=zeros(n,1); //initializing the 1D data
loc_count=k/q; //no of locations where we place clusters
locations=tau+1+pmodulo(ceil(rand(loc_count,1)*1000),(n-2*tau)); //generating locations to cluster
//Populating chosen locations with +-1 generated randomly
for i=1:size(locations,1)
    loc_temp=locations(i);
    for j=(loc_temp-tau):(loc_temp+tau)
        if rand(1)>0.5 then
            x(j)=1;
        else
            x(j)=-1;
        end
    end
end


//Generating measurement noise
v=rand(m,1,'normal')*sigma; //gaussian noise having zero mean and sd=sigma

//Genrating measurements
y=(phi*x)+v;

//plot(phi*x);
//plot(y,'r');

//DGS approximation pruning algorihtm
//with input x,k,N_x,w and tau
//and output supp_x
function supp_x=DGS_approx_prune(x,k,N_x,w,tau)
    z=0;
    for i=1:n
        sum_nx=0;
        for t=1:tau
            w_temp=w(i,t)^2;
            N_x_temp=N_x(i,1)^2;
            sum_nx=sum_nx+(w_temp*N_x_temp);
        end
        z(i)=(x(i)^2)+sum_nx;
    end
    z_sorted=gsort(z,'g','i');
    supp_x=z_sorted(1:k);
endfunction

//AdaDGS recovery algorithm
//input is phi,y,k_min,k_max,del_k
//output is x_hat
function x_hat=AdaDGS_recovery(phi,y,k_min,k_max,del_k)
    //initialization
    y_r=y; //y_residue
    k=k_min;
    supp_x=zeros(k); //support of x
    x_hat=0; 
    
    
endfunction
    
