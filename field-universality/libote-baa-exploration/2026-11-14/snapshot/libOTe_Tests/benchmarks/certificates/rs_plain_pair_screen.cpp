// Floating evaluation of the finite pair-rank upper chain.
// Diagnostic only: not a rigorously rounded probability certificate.
// Arguments: source weight, dimension, pair returns, tau.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
int main(int argc,char**argv){
 if(argc!=5 && argc!=6)return 2;
 const size_t s=std::stoull(argv[1]),D=std::stoull(argv[2]),pairs=std::stoull(argv[3]);
 const double tau=std::stod(argv[4]);
 const bool minimumOnly=argc==6;
 if(minimumOnly && s!=9*D)return 2;
 if(!D||2*pairs>s||tau<1)return 2;
 std::vector<double> a(D),b(D);a[0]=1;double logscale=0;
 for(size_t j=0;j<pairs;++j){
  const size_t m=s-2*j,active=std::min(D,j+2);double prev=0,total=0;
  const double inv=1.0/(m-1);
  for(size_t r=0;r<active;++r){
   double rho=std::min(1.0,(3*r+2)*inv);
   if(minimumOnly)rho=std::min(rho,(15.0*r*(r+1)+6.0*D)*inv/m);
   b[r]=a[r]*rho+prev;
   prev=a[r]*(1-rho)/tau;
   total+=b[r];
  }
  if(!(total>0))return 3;
  logscale+=std::log(total);
  for(size_t r=0;r<active;++r)b[r]/=total;
  a.swap(b);
  if((j+1)%10000==0||j+1==pairs)
   std::cout<<s<<','<<D<<','<<j+1<<','<<tau<<','<<std::setprecision(15)<<logscale/std::log(2)<<std::endl;
 }
}
