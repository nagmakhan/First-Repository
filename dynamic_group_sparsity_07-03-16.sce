//Implmenting algorithm in Huang-Learning with dynamic group sparsity
//1D case
abort
clear
//Defining constants
n=512; //size of data
k=64; //sparsity
//q=4; //no of clusters
//n=64; //size of data
//k=8; //sparsity
//k=3;
q=4; //no of clusters
//m=6*k; //no of measurements
fac=ceil(log(n/k));
m=(fac)*k; //no of measurements
sigma=0.01; //measurement noise
tau=2; //no of neighbours
//tau=0;
k_min=1;
k_max=n/3; //range of k
steps=50; //steps to increase k
del_k=round((k_max-k_min)/steps); //step size for increasing k
w_init=0.5; //to initialise the weights
w=w_init*ones(n,tau); //weights
N_x=zeros(n,tau);//neighbours value
//epsilon=(10^-1)*zeros(n,1); //tolerance level
epsilon=0.01;

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
x_meas=zeros(n,1); //initializing the 1D data
//x_meas(2:66)=1;
//x_meas(2:33)=1;
//x_meas(50:81)=-1;
//x(2)=1
//x(34)=4;
//x(50)=-1;
//x(4)=5;
//x(21)=1;
//x(30)=4;
//x(25)=-1;
//x(40)=5;
loc_count=q; //no of locations where we place clusters
disp(loc_count)
//locations=tau+1+pmodulo(ceil(rand(loc_count,1)*1000),(n-2*tau)); //generating locations to cluster
locations=pmodulo(ceil(rand(loc_count,1)*1000),(n)); //generating locations to cluster
disp(locations)
//Populating chosen locations with +-1 generated randomly
for i=1:size(locations,1)
    loc_temp=locations(i);
    if loc_temp>=(n-(k/q)) then
        loc_temp=loc_temp-(k/q)+1;
    end
    for j=(loc_temp):(loc_temp+(k/q)-1)
        if rand(1)>0.5 then
            x_meas(j)=1;
        else
            x_meas(j)=-1;
        end
    end
end
//plot(x_meas)
//x_sorted=gsort(x,'g','d');
//Generating measurement noise
v=rand(m,1,'normal')*sigma; //gaussian noise having zero mean and sd=sigma
//v=0;
//Genrating measurements
y=(phi*x_meas)+v;

//plot(phi*x);
//plot(y,'r');
//make x a row vector
function x_row=row(x)
    if size(x,2)~=1 then
        x=x';
    end
    x_row=x;    
endfunction

//function to find support 
function supp=find_support(x,k)
    [x_temp,index]=gsort(row(abs(x)),'g','d');
    supp=index(1:k); //selecting k largest elements
    //supp=gsort(supp,'g','i'); //sorting the indices
    supp=row(supp);
endfunction

//find non-zero components
function supp=support(x)
    x=row(x);
    supp=zeros(size(x,1));
    supp=find(abs(x)>0.01); //finding non-zero elements
endfunction

//to get phi_T
function phi_T=return_phi_T(phi,T)
    T=row(T); //make T a row vector
    row_T=size(T,1); //no fo rows in T
    phi_T=phi(:,T);
endfunction

//function to merge sets
function merged_supp=merge_support(supp1,supp2)
    if supp1(1)~=0 & supp2(1)~=0 then
        merged_supp_temp=[row(supp1);row(supp2)];
    elseif supp1(1)==0 then
        merged_supp_temp=[row(supp2)];
    elseif supp2(1)==0 then
        merged_supp_temp=[row(supp1)];
    else
        disp('Merged support set is empty!')
    end
    merged_supp=row(unique(merged_supp_temp));
endfunction

//estimating
function x=estimate(phi,y,T)
    [r,c]=size(phi);
    x=zeros(c,1);
    phi_T=return_phi_T(phi);
    x(T)=phi_T\y;
endfunction

//DGS approximation pruning algorihtm
//with input x,k,N_x,w and tau
//and output supp_x
function supp_x=DGS_approx_prune(x,k,N_x,w,tau)
    z=0;
    for i=1:n
        sum_nx=0;
        for t=1:tau
            w_temp=w(i,t)^2;
            N_x_temp=N_x(i,t)^2;
            sum_nx=sum_nx+(w_temp*N_x_temp);
        end
        z(i)=(x(i)^2)+sum_nx;
    end
    [z_sorted,index]=gsort(abs(z),'g','d');
    supp_x=index(1:k);
endfunction

//function to make a vector k-sparse
function x_sparse=make_sparse(x,k)
    x=row(x); //make x a row vector
    row_x=size(x,1); //no of rows in x
	index=DGS_approx_prune(x,k,N_x,w,tau);
    x_sparse=zeros(row_x,1);
    x_sparse(index)=x(index);
endfunction

//AdaDGS recovery algorithm
//input is phi,y,k_min,k_max,del_k
//output is x_hat
function x_hat=AdaDGS_recovery(phi,y,k)
    //initiliazation
    [phi_row,phi_col]=size(phi); //size of x
    x_hat_vec=zeros(phi_col,1); //to store estimate of recovered signal in each iteration
    x_hat_temp=[x_hat_vec]; //storing the x_hat_vec
    y_r=y;  //to store residue
    difference=100;
    //flag=1; //to keep track of halting criterion
    i=1; //index
    //w=w_init*ones(n,tau);
    while abs(difference)>epsilon
         i=i+1;
        //merging support sets
        x_temp=phi'*y_r;
        supp_xtemp=DGS_approx_prune(x_temp,2*k,N_x,w,tau);
        supp_x_hat=support(x_hat_vec);
        merged_supp=merge_support(supp_xtemp,supp_x_hat);
        //estimating x by LS
        b=estimate(phi,y,merged_supp); //estimate of x
        //prune to obtain next estmate
        x_hat_vec=make_sparse(b);
        x_hat_temp=[x_hat_temp x_hat_vec];
        //update current samples
        y_r=y-(phi*x_hat_vec);
        //check halting criterion
        difference=max(abs(x_hat_temp(:,i)-x_hat_temp(:,i-1)));
        //difference=abs(x_hat_temp(:,i)-x_meas);
        disp(difference)
        disp(i)
        //if difference<epsilon then
        //    flag=0;
        //end
        x_hat=x_hat_temp(:,i);
    end
endfunction
    
x_hat=AdaDGS_recovery(phi,y,k);
//plot(x_hat(:,$),'r')
//plot(x)
n_vec=1:n;
plot2d3(n_vec,[x_meas x_hat])
figure,plot(x_meas-x_hat)
