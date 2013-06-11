#include <iostream>
#include <vector>
#include <exception>
#include <boost/random.hpp>
#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <RcppArmadillo.h>
#define NDEBUG
#include <cppbugs/cppbugs.hpp>
#include <cppbugs/distributions/mcmc.wishart.hpp>
#include <cppbugs/distributions/mcmc.mvcar.hpp>
#include <cppbugs/distributions/mcmc.poisson.hpp>
//#include <cppbugs/deterministics/mcmc.linear.grouped.hpp>
//#include <cppbugs/deterministics/mcmc.linear.hpp>
//#include <cppbugs/deterministics/mcmc.rsquared.hpp>
#include <cppbugs/deterministics/mcmc.lambda.hpp>
#include <cppbugs/mcmc.model.hpp>

using namespace arma;
using namespace cppbugs;
using std::cout;
using std::endl;

Cube<double> update_mrate(const vec& b0, const arma::Mat<double>& b1, const arma::Mat<double>& b2, const arma::Mat<double>& b3) {
  Cube<double> ans;
  for(int s = 0; s < b0.n_elem; ++s) {
    for(int i = 0; i < b2.n_cols; ++i) {
      for(int x = 0; x < b1.n_cols; ++x) {
        ans[s,i,x] <- b0[s] + b1[s,x] + b2[s,i]*b3[s,x];
      }
    }
  }
  return 1/(1+exp(-ans));
}

Cube<double> update_prob(const vec& b4, const mat& b5, const mat& b6, const mat& b7) {
  Cube<double> ans;
  for(int s = 0; s < b4.n_elem; ++s) {
    for(int i = 0; i < b6.n_cols; ++i) {
      for(int x = 0; x < b5.n_cols; ++x) {
        ans[s,i,x] <- b4[s]  + b5[s,x]  + b6[s,i]*b7[s,x];
      }
    }
  }
  // FIXME -- use logit
  return 1/(1+exp(-ans));
}


mat normalize_by_rowsum(const mat& x) {
  const uvec z(zeros<uvec>(x.n_cols));
  colvec rs = sum(x,1);
  return x / rs.cols(z);
}

SEXP run_hle(const Cube<double>& pop, const Cube<double>& deaths, const Cube<double>& nr_healthy, const Cube<double>& nr_resp, const vec& adj_geo, const vec& numNeigh_geo, const int sumNumNeigh_geo,  vec b0,  mat b1,  mat b2,  mat f3,  vec b4,  mat b5,  mat b6,  mat f7, mat tau_b1,  mat tau_b2,  mat tau_b5,  mat tau_b6) {
  // sex  s = 0,1
  // area i = 0..(N-1)
  // ages x = 0..(X-1)
  // survey age groups = 0..(M-1)
  int X = pop.n_slices; // (19)
  int N = pop.n_cols; // (28)
  int S = pop.n_rows; // (2)
  const int M(15);
  mat Q; Q.eye(2,2);
  Cube<double> gamma(S,N,X);
  // dim of deaths = (S,N,X)
  Cube<double> mrate(S,N,X);
  //Cube<double> nr_healthy(S,N,M);
  Cube<double> prob(S,N,M);
  //Cube<double> nr_resp(S,N,M);

  mat b3(S,X);  // deterministic linked to f3
  mat b7(S,M);  // deterministic linked to f7

  vec weight_geo(sumNumNeigh_geo);
  weight_geo.fill(1);

  // length of age intervals
  ivec n(X);
  n[0] = 1;
  n[1] = 4;
  for(int i = 2; i < X; ++i) { 
    n[i] = 5;
  }

  // proportion of interval n[x] lived by those who die in the interval
  vec a(X);
  a[0] = 0.1;
  for(int i = 1; i < X; ++i) {
    a[i] = 0.5;
  }

  // B1:
  // these are calculated but const for the whole run
  // adj, wgts, neighbors
  vec adj_b1(X);
  vec weight_b1(X);
  vec numNeigh_b1(X);

  adj_b1[0] = 2;
  weight_b1[0] = 1;
  numNeigh_b1[0] = 1;

  for(int i = 1; i < X; ++i) {
    adj_b1[2+(i-1)*2] = i-1;
    adj_b1[3+(i-1)*2] = i+1;
    weight_b1[2+(i-1)*2] = 1;
    weight_b1[3+(i-1)*2] = 1;
    numNeigh_b1[i] = 2;
  }
  weight_b1[(X-2)*2 + 2] = 1;
  adj_b1[(X-2)*2 + 2] = X-1;
  numNeigh_b1[X-1] = 1;


  // B5:
  // these are calculated but const for the whole run
  // adj, wgts, neighbors

  vec adj_b5(M);
  vec weight_b5(M);
  vec numNeigh_b5(M);

  adj_b5[0] = 2;
  weight_b5[0] = 1;
  numNeigh_b5[0] = 1;

  for(int i = 1; i < M; ++i) {
    adj_b5[2+(i-1)*2] = i-1;
    adj_b5[3+(i-1)*2] = i+1;
    weight_b5[2+(i-1)*2] = 1;
    weight_b5[3+(i-1)*2] = 1;
    numNeigh_b5[i] = 2;
  }
  weight_b5[(M-2)*2 + 2] = 1;
  adj_b5[(M-2)*2 + 2] = M-1;
  numNeigh_b5[M-1] = 1;

  MCModel<boost::minstd_rand> m;

  //b0[s]  ~ dflat()
  m.link<Uniform>(b0,-1e6,1e6);
  //b4[s]  ~ dflat()
  m.link<Uniform>(b4,-1e6,1e6);

  // b1[1:2,1:X]  ~ mv.car(adj_b1[], weight_b1[], numNeigh_b1[], tau_b1[,] )
  // tau_b1[1:2 ,1:2] ~dwish(Q[,] ,2)
  m.link<MvCar>(b1, adj_b1, weight_b1, numNeigh_b1, tau_b1);
  m.link<Wishart>(tau_b1, Q, 2);

  // b2 prior -> note that adj_geo[], numNeigh_geo[], and sumNumNeigh_geo are loaded as data.
  // b2[1:2,1:N]  ~ mv.car(adj_geo[],weight_geo[],numNeigh_geo[],tau_b2[,] )
  // tau_b2[1:2,1:2] ~ dwish(Q[,],2)
  m.link<MvCar>(b2, adj_geo, weight_geo, numNeigh_geo, tau_b2);
  m.link<Wishart>(tau_b2, Q, 2);

  // b5 prior
  // b5[1:2,1:M]  ~ mv.car(adj_b5[], weight_b5[], numNeigh_b5[], tau_b5[,] )
  // tau_b5[1:2 ,1:2] ~dwish(Q[,] ,2)
  m.link<MvCar>(b5, adj_b5, weight_b5, numNeigh_b5, tau_b5);
  m.link<Wishart>(tau_b5, Q, 2);

  // b6 prior -> note that adj_geo[], numNeigh_geo[], and sumNumNeigh_geo are loaded as data.
  // b6[1:2,1:N]  ~ mv.car(adj_geo[],weight_geo[],numNeigh_geo[],tau_b6[,] )
  // tau_b6[1:2,1:2] ~dwish(Q[,],2)
  m.link<MvCar>(b6, adj_geo, weight_geo, numNeigh_geo, tau_b6);
  m.link<Wishart>(tau_b6, Q, 2);


  // # b3 & b7  prior 
  // for (s in 1:2){
  //  for (x in 1:X){
  //       b3[s,x] <- f3[s,x]/sum(f3[s,])
  //       f3[s,x] ~ dgamma(1,1)
  //  }
  //  for (x in 1:M){ 
  //   b7[s,x] <- f7[s,x]/sum(f7[s,])
  //   f7[s,x] ~ dgamma(1,1)
  // }}
  m.link<Gamma>(f3,1,1);
  m.lambda<mat,mat>(b3,normalize_by_rowsum,f3);
  m.link<Gamma>(f7,1,1);
  m.lambda<mat,mat>(b7,normalize_by_rowsum, f7);

  // # poisson link function
  // for (x in 1:X){ 
  //   log(mrate[s,i,x]) <- b0[s] + b1[s,x] + b2[s,i]*b3[s,x] 
  // } 
  m.lambda<Cube<double>, vec, mat, mat, mat>(mrate, update_mrate, b0,b1,b2,b3);

  // # poisson likelihood
  // for (x in 1:X){ 
  //  gamma[s,i,x] <- pop[s,i,x] * mrate[s,i,x]
  //  deaths[s,i,x] ~ dpois(gamma[s,i,x])
  // }
  m.lambda<Cube<double>, Cube<double>, Cube<double> >(gamma, [](const Cube<double>& x, const Cube<double>& y) { return x % y; }, pop, mrate);
  m.link<ObservedPoisson>(deaths,gamma);


  // # binomial link function
  // for (x in 1:M){ 
  //   logit(prob[s,i,x])  <- b4[s]  + b5[s,x]  + b6[s,i]*b7[s,x]
  // } 
  m.lambda<Cube<double>,vec, mat, mat, mat>(prob,update_prob,b4,b5,b6,b7);

  // # binomial likelihood
  // for (x in 1:M){   
  //    nr.healthy[s,i,x] ~ dbin(prob[s,i,x], nr.resp[s,i,x])
  // }  
  m.link<ObservedBinomial>(nr_healthy,prob,nr_resp);


  std::vector<arma::vec>& b0_hist = m.track<std::vector>(b0);
  std::vector<arma::mat>& b1_hist = m.track<std::vector>(b1);
  std::vector<arma::mat>& b2_hist = m.track<std::vector>(b2);
  std::vector<arma::mat>& b3_hist = m.track<std::vector>(b3);
  std::vector<arma::vec>& b4_hist = m.track<std::vector>(b4);
  std::vector<arma::mat>& b5_hist = m.track<std::vector>(b5);
  std::vector<arma::mat>& b6_hist = m.track<std::vector>(b6);

  m.tune(1e4,100);
  m.tune_global(1e4,100);
  m.burn(1e4);
  m.sample(1e5, 10);

  cout << "b: " << endl << mean(b0_hist.begin(),b0_hist.end()) << endl;
  cout << "samples: " << b0_hist.size() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;

  return R_NilValue;
}

const Cube<double> toCube(SEXP x) {
  if(TYPEOF(x)!=REALSXP) {
    throw std::logic_error("not a real valued array.");
  }
  SEXP dim_ = Rf_getAttrib(x,R_DimSymbol);
  if(Rf_length(dim_) != 3) {
    throw std::logic_error("not a cube.");
  }
  return cube(REAL(x),INTEGER(dim_)[0],INTEGER(dim_)[1],INTEGER(dim_)[2],false,true);
}

extern "C" SEXP hle(SEXP pop_, SEXP deaths_, SEXP nr_healthy_, SEXP nr_resp_, SEXP adj_geo_, SEXP numNeigh_geo_, SEXP sumNumNeigh_geo_,  SEXP b0_,  SEXP b1_,  SEXP b2_,  SEXP f3_,  SEXP b4_,  SEXP b5_,  SEXP b6_,  SEXP f7_, SEXP tau_b1_,  SEXP tau_b2_,  SEXP tau_b5_,  SEXP tau_b6_) {

  const Cube<double> pop(toCube(pop_));
  const Cube<double> deaths(toCube(deaths_));
  const Cube<double> nr_healthy(toCube(nr_healthy_));
  const Cube<double> nr_resp(toCube(nr_resp_));
  const vec adj_geo(Rcpp::as<vec>(adj_geo_));
  const vec numNeigh_geo(Rcpp::as<vec>(numNeigh_geo_));
  const int sumNumNeigh_geo(Rcpp::as<int>(sumNumNeigh_geo_));

  vec b0 = Rcpp::as<vec>(b0_);
  mat b1 =  Rcpp::as<mat>(b1_);
  mat b2 = Rcpp::as<mat>(b2_);
  mat f3 = Rcpp::as<mat>(f3_);
  vec b4 = Rcpp::as<vec>(b4_);
  mat b5 = Rcpp::as<mat>(b5_);
  mat b6 = Rcpp::as<mat>(b6_);
  mat f7 = Rcpp::as<mat>(f7_);

  mat tau_b1 = Rcpp::as<mat>(tau_b1_);
  mat tau_b2 = Rcpp::as<mat>(tau_b2_);
  mat tau_b5 = Rcpp::as<mat>(tau_b5_);
  mat tau_b6 = Rcpp::as<mat>(tau_b6_);

  // FIXME: checks for matching dimensions of coefs to data...

  return run_hle(pop, deaths, nr_healthy, nr_resp, adj_geo, numNeigh_geo, sumNumNeigh_geo,  b0, b1, b2, f3, b4, b5, b6, f7, tau_b1, tau_b2, tau_b5, tau_b6);
}
