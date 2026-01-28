#pragma once


#include <functional>


template<typename T, class Gauge> // T currently for double only
struct Obs {
  std::string description;
  double param;
  std::function<T(const Gauge&)> f;
  T exact;

  std::vector<T> v;

  Obs
  (
   const std::string& description_,
   const double param,
   const std::function<T(const Gauge&)>& f_,
   const T exact_
   )
    : description(description_)
    , param(param)
    , f(f_)
    , exact(exact_)
  {
    v.clear();
  }

  void meas( const Gauge& W ) { v.push_back( f(W) ); }

  T mean() const {
    assert( v.size()!=0 );
    T res = 0.0;
    int counter=0;
    for(const auto elem : v) {
      res += elem;
      counter++;
    }
    res /= counter;
    return res;
  }

  T err( const T& mean_, const bool is_sample_mean=true ) const {
    assert( v.size()!=0 );
    T res = 0.0;
    int counter=0;
    for(const auto elem : v) {
      res += (elem-mean_)*(elem-mean_);
      counter++;
    }
    res /= counter;

    if(is_sample_mean) res /= (counter-1);
    else res /= counter;
    res = std::sqrt( res );

    return res;
  }

  void jackknife( T& mean, T& err,
		  const int binsize = 1 ) const {
    const int nbins = v.size() / binsize;
    const int N = binsize * nbins;
    std::vector<T> binned;
    std::vector<T> jack_avg;

    auto iv = v.begin();
    for(int j=0; j<nbins; j++){
      T avg = 0.0;
      for(int k=0; k<binsize; k++){
	avg += *iv;
	iv++;
      }
      avg /= binsize;
      binned.push_back( avg );
    }

    for(int j=0; j<nbins; j++){
      T avg = 0.0;
      for(int k=0; k<nbins; k++){
	if(j==k) continue;
	avg += binned[k];
      }
      avg /= nbins-1;
      jack_avg.push_back( avg );
    }

    // -------------------

    mean = 0.0;
    for(int j=0; j<nbins; j++) mean += jack_avg[j];
    mean /= nbins;

    double var = 0.0;
    for(int j=0; j<nbins; j++) var += (mean-jack_avg[j])*(mean-jack_avg[j]);
    var /= nbins;
    var *= nbins-1;
    err = std::sqrt(var);
  }
  
};
