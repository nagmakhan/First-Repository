//Implmenting algorithm in Huang-Learning with dynamic group sparsity
//1D case
//This code deals with all real values

clear;
//Defining constants
//n=512; //size of data
//k=64; //sparsity
//q=4; //no of clusters
n=64; //size of data
k=8; //sparsity
q=2; //no of clusters
//m=3*k; //no of measurements
m=3*k; //no of measurements
sigma=0.01; //measurement noise
tau=2; //no of neighbours
k_min=1;
k_max=n/3; //range of k
steps=50; //steps to increase k
del_k=round((k_max-k_min)/steps); //step size for increasing k
//epsilon=(10^-1)*zeros(n,1); //tolerance level
epsilon=(10^-1);

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

x_sorted=gsort(x,'g','d');
//Generating measurement noise
v=rand(m,1,'normal')*sigma; //gaussian noise having zero mean and sd=sigma

//Genrating measurements
y=(phi*x)+v;

//plot(phi*x);
//plot(y,'r');
//make x a row vector
function x_row=row(x)
    if size(x,2)~=1 then
        x=x';
    end
    x_row=x;    
endfunction

//function to prune
//same as make_sparse(x,k) function
//function y_pruned=prune(y,k)
//    y=row(y); //making y a row vector
//    [y_pruned_temp index]=gsort(y,'g','d'); //sorting in descending order
//    for i=1:size(y,1)
//        for j=1:k
//            if i==index(j) then
//                
//            end
//        end
//    end
//    y_pruned=y_pruned(1:k);
//endfunction
//
//function to find support 
function supp=find_support(x,k)
    [x_temp,index]=gsort(row(x),'g','d');
    supp=index(1:k); //selecting k largest elements
    //supp=gsort(supp,'g','i'); //sorting the indices
    supp=row(supp);
endfunction

//find non-zero components
function supp=support(x)
    x=row(x);
    supp=zeros(size(x,1));
    [x_temp,index]=gsort(x,'g','d');
    for i=1:size(x_temp,1)
        if x_temp(i)~=0 then
            supp(i)=index(i);
        else
            break;
        end
    end
    supp_temp=gsort(supp,'g','d'); //sorting the indices
    if supp_temp(1)==0 then
        disp('Warning:support set is empty!')
    end
    supp=row(gsort(supp,'g','i'));
    //sort so as not to change order of elments
endfunction

////to get phi_T
//function phi_T=return_phi_T(phi,T)
//    T=row(T); //make T a row vector
//    row_T=size(T,1); //no fo rows in T
//    [row_phi,col_phi]=size(phi);
//    phi_T=zeros(row_phi,col_phi); //initialiazing phi_T
//    [T_sort]=(gsort(T,'g','i')); //sort in ascending order
//    for i=1:col_phi
//        for j=1:row_T
//            if i==T_sort(j) then
//                phi_T(:,i)=phi(:,i);
//            end
//        end
//    end
//   // phi_T=phi_T_temp;
//endfunction

//to get phi_T
function phi_T=return_phi_T(phi,T)
    T=row(T); //make T a row vector
    row_T=size(T,1); //no fo rows in T
    //[row_phi,col_phi]=size(phi);
    //phi_T=zeros(row_phi,col_phi); //initialiazing phi_T
    //[T_sort]=(gsort(T,'g','i')); //sort in ascending order->no need
    //as support set will be sorted always in our case
    for i=1:row_T
        phi_T(:,i)=phi(:,T(i));
    end
   // phi_T=phi_T_temp;
endfunction

//T=[1 3 4]
//temp=return_phi_T(phi,T)

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

//x=[1 0 4 5 0]
//supp=support(x)
//y=[2 0 4 0 0]
//supp2=support(y)
//supp_m=merge_support(supp,supp2)

//estimating
function x=estimate(phi,y,T)
    phi_T=return_phi_T(phi,T);
    x=inv(phi_T'*phi_T)*phi_T'*y;
endfunction

//function to make a vector k-sparse
function x_sparse=make_sparse(x,k)
    x=row(x); //make x a row vector
    row_x=size(x,1); //no of rows in x
    [x_temp,index]=gsort(x,'g','d'); //sorting x in descending order
    index=index(1:k);
    x_sparse=zeros(row_x,1);
    for i=1:row_x
        for j=1:k
            if(i==index(j))
                x_sparse(i)=x(i);
                break
            end
        end
    end
endfunction

//CoSaMP
//It is assumed that sparsity is known
function x_hat=CoSaMP(phi,y,k)
    //initiliazation
    [phi_row,phi_col]=size(phi); //size of x
    //disp(phi_col)
    x_hat_vec=zeros(phi_col,1); //to store estimate of recovered signal in each iteration
    //x_hat_temp=[];
    x_hat_temp=[x_hat_vec]; //storing the x_hat_vec
    y_r=y;  //to store residue
    //k=k_min; //sparsity
    flag=1; //to keep track of halting criterion
    i=1; //index
    while flag==1
        i=i+1;
        //k=k_min+del_k;
        
        //merging support sets
        x_temp=phi'*y_r;
        supp_xtemp=find_support(x_temp,2*k);
        supp_x_hat=support(x_hat_vec);
        merged_supp=merge_support(supp_xtemp,supp_x_hat);
        
        //estimating x by LS
        b=estimate(phi,y,merged_supp); //estimate of x
        //disp(size(b))
        b=[b;zeros(phi_col-size(b,1),1)]; //making b the size of x
        //prune to obtain next estmate
        x_hat_vec=make_sparse(b);
//        disp(i)
//        disp(size(b))
//        disp(size(x_hat_vec))
//        disp(size(x_hat_temp))
        [x_hat_temp2,index]=gsort(row(b),'g','d');
//        x_hat_vec=row(x_hat_temp2(1:k));
        x_hat_temp=[x_hat_temp x_hat_vec];
        
        //update current samples
        //phi_T=return_phi_T(phi,index(1:k));
        y_r=y-(phi*x_hat_vec);
        
        //check halting criterion
        difference=max(x_hat_temp(:,i)-x_sorted);
        //disp(difference)
        disp(i)
        if i>500 then
            flag=0;
        end
//        if abs(difference)<=epsilon then
//            flag=0;
//        end
    end
    x_hat=x_hat_temp(:,i);
    //disp(size(x_hat))
    //temp=zeros(n-k,1);
    //disp(size(temp))
    //x_hat=[x_hat;temp];
endfunction

x_hat=CoSaMP(phi,y,k);
plot(x_hat,'r')
plot(x)
