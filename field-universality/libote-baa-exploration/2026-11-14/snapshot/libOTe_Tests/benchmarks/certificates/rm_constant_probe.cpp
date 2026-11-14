// Exact small-field diagnostic. It does not certify GF(2^128) distance.
// Build: c++ -O3 -std=c++20 rm_constant_probe.cpp -o rm_constant_probe
#include <array>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using U=uint32_t;
constexpr std::array<unsigned,16> mons{0,1,2,3,4,5,6,8,9,10,12,16,17,18,20,24};
using Vec=std::array<unsigned,32>;
using Perm=std::array<unsigned,32>;
unsigned mul(unsigned a,unsigned b,unsigned modulus) {
 unsigned r=0,top=1u<<(std::bit_width(modulus)-1);
 while(b){if(b&1)r^=a;b>>=1;a<<=1;if(a&top)a^=modulus;}return r;
}
unsigned inv(unsigned a,unsigned mod) {
 unsigned r=1,q=1u<<(std::bit_width(mod)-1),e=q-2;
 while(e){if(e&1)r=mul(r,a,mod);a=mul(a,a,mod);e>>=1;}return r;
}
Vec stage(Vec x,const Perm& p,unsigned c,unsigned mod) {
 Vec y{};unsigned s=0;for(unsigned i=0;i<32;++i)y[i]=s=x[p[i]]^mul(c,s,mod);return y;
}
Vec unstage(Vec y,const Perm& p,unsigned c,unsigned mod) {
 Vec x{};for(unsigned i=0;i<32;++i)x[p[i]]=y[i]^(i?mul(c,y[i-1],mod):0);return x;
}
Vec local(unsigned j) {
 Vec v{};for(unsigned i=0;i<32;++i)v[i]=((i&mons[j])==mons[j]);return v;
}
using Col=std::array<unsigned,16>;
using Matrix=std::array<Col,32>;
Matrix parity(const Perm& p,const Perm& q,unsigned c,unsigned mod) {
 Matrix h{};
 for(unsigned i=0;i<32;++i){Vec e{};e[i]=1;auto x=unstage(unstage(e,q,c,mod),p,c,mod);
  for(unsigned j=0;j<16;++j)for(unsigned t=0;t<32;++t)if((t&mons[j])==mons[j])h[i][j]^=x[t];}
 return h;
}
struct Basis {std::array<Col,16> rows{};};
bool insert(Basis& b,Col v,unsigned mod) {
 for(unsigned i=0;i<16;++i)if(v[i]){
  if(b.rows[i][i]){unsigned a=v[i];for(unsigned j=i;j<16;++j)v[j]^=mul(a,b.rows[i][j],mod);}
  else {unsigned a=inv(v[i],mod);for(unsigned j=i;j<16;++j)b.rows[i][j]=mul(a,v[j],mod);return true;}
 }return false;
}
bool search(const Matrix& h,unsigned mod,unsigned target,unsigned begin,const Basis& basis,
            std::vector<unsigned>& support,uint64_t& visits,uint64_t limit) {
 for(unsigned j=begin;j+target<=32;++j){
  if(++visits>limit)return false;
  Basis next=basis;support.push_back(j);
  if(!insert(next,h[j],mod))return true;
  if(target>1 && search(h,mod,target-1,j+1,next,support,visits,limit))return true;
  support.pop_back();
 }return false;
}
int main(int argc,char**argv) {
 unsigned trials=argc>1?std::stoul(argv[1]):100,mod=argc>2?std::stoul(argv[2]):7;
 uint64_t limit=argc>3?std::stoull(argv[3]):2000000;
 std::mt19937 gen(4127);uint64_t visits=0;
 for(unsigned trial=0;trial<trials;++trial){
  Perm p,q;std::iota(p.begin(),p.end(),0);std::iota(q.begin(),q.end(),0);
  std::shuffle(p.begin(),p.end(),gen);std::shuffle(q.begin(),q.end(),gen);
  std::array<U,16> g{};unsigned db=32;U word=0;
  for(unsigned j=0;j<16;++j){auto v=stage(stage(local(j),p,1,mod),q,1,mod);
   for(unsigned i=0;i<32;++i)g[j]|=U(v[i])<<i;}
  for(U i=1;i<65536;++i){word^=g[std::countr_zero(i)];db=std::min(db,unsigned(std::popcount(word)));}
  auto h=parity(p,q,2,mod);Basis basis;std::vector<unsigned> support;uint64_t before=visits;
  bool found=search(h,mod,db-1,0,basis,support,visits,visits+limit);
  if(found){
   std::cout<<"FOUND trial="<<trial<<" modulus="<<mod<<" binary_distance="<<db<<" mixed_support="<<support.size()<<"\n";
   for(auto* a:{&p,&q}){for(auto x:*a)std::cout<<x<<',';std::cout<<'\n';}
   for(auto x:support)std::cout<<x<<',';std::cout<<'\n';return 0;
  }
  if(trial%10==0)std::cout<<"trial="<<trial<<" binary_distance="<<db<<" visited="<<visits-before<<'\n'<<std::flush;
 }
 std::cout<<"NO WITNESS in diagnostic search; not a proof. visits="<<visits<<'\n';
}
