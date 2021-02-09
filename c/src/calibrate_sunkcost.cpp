#include <initializer_list>
#include <iostream>
#include <utility>
#include <cstring>
#include <fstream>
#include <sstream>
#include <time.h>
#include <omp.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_types.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/de1220.hpp>
#include <pagmo/algorithms/nlopt.hpp>
#include <pagmo/types.hpp>

int fcnt;

extern "C" double kappa0;
extern "C" double kappa1;

extern "C" void linebreak();
extern "C" void linebreak2();
extern "C" int init_params();
extern "C" void discretize_x(int pareto);
extern "C" void discretize_z();
extern "C" void calc_expected_productivity();
extern "C" void random_draws();
extern "C" int solve_policies_all_dests();
extern "C" void simul_all_dests(double * expart_rate, double * exit_rate);

double fitness_func(double params[2])
{
  time_t start, stop;
  time(&start);

  fcnt++;
  
  kappa0=params[0];
  kappa1=params[1];
  
  printf("\nCandidate parameter vector %d:\n",fcnt);
  printf("\tkappa0 = %0.8f\n",kappa0);
  printf("\tkappa1 = %0.8f\n",kappa1);
  
  if(solve_policies_all_dests())
    return GSL_NAN;
    
  double expart_rate=0.0;
  double exit_rate=0.0;
  simul_all_dests(&expart_rate,&exit_rate);

  return sqrt(expart_rate*expart_rate + exit_rate*exit_rate);
}

using namespace pagmo;

// Objective function: utilitarian welfare criterion subject to box bounds
struct PAGMO_DLL_PUBLIC calibrate_sunkcost
  {
   // empty constuctor
   calibrate_sunkcost(){};

   // problem name
   std::string get_name() const
   {
     return "Sunk cost model calibration objective function";
   }

   // Implementation of the objective function.
   vector_double fitness(const vector_double &dv) const
   {
     double params[2];
     for(int i=0; i<2; i++)
       {
	 params[i] = dv[i];
       }
     return {fitness_func(params)};
   }
  
   // Implementation of the box bounds.
   std::pair<vector_double, vector_double> get_bounds() const
   {
     return {{0.5, 0.05},{3.0,1.0}};

   template <typename Archive>
   void serialize(Archive &ar, unsigned){}
  };

PAGMO_S11N_PROBLEM_EXPORT_KEY(calibrate_sunkcost)
PAGMO_S11N_PROBLEM_IMPLEMENT(calibrate_sunkcost)

int main(int argc, char * argv[])
{
  time_t start, stop;
  time(&start);

  printf("\nExport Market Penetration Dynamics, Joseph Steinberg, University of Toronto\n\n");

#ifdef _OPENMP
  printf("Parallel processing with %d threads\n",omp_get_max_threads());
#else
  printf("No parallelization\n");
#endif

  if(init_params())
    {
      printf("Failed to initialize parameters!\n");
      return 1;
    }

  discretize_x(0);
  discretize_z();
  calc_expected_productivity();
  random_draws();
  
  fcnt=0;
  
  problem prob{calibrate_sunkcost{}};
  std::cout << "\n" << prob << '\n';
  
  algorithm algo{de1220(100)};
  algo.set_verbosity(1);
  std::cout << "\n" << algo << '\n';
  
  linebreak();
  
  population pop{prob, 48};
  //population pop{problem{}};
  //std::ifstream ifs("output/calibration_population.txt");
  //boost::archive::text_iarchive ia(ifs);
  //pop.load(ia,1);

  //std::cout << "\n" << pop << '\n';
  //linebreak();
    
  pop = algo.evolve(pop);
  
  linebreak();
  std::cout << "\nChampion:\n";
  for(int i=0; i<12; i++)
    {
      std::cout << pop.champion_x()[i] << "\n";
    }

  std::ofstream ofs("output/sunkcost_population.txt");
  boost::archive::text_oarchive oa(ofs);
  pop.save(oa,1);

  linebreak();

  nlopt local("neldermead");
  local.set_selection("best");
  local.set_replacement("best");
  local.set_maxeval(120);
  local.set_verbosity(1);

  linebreak();
  
  pop = local.evolve(pop);

  linebreak();
  std::cout << "\nNew champion:\n";
  for(int i=0; i<12; i++)
    {
      std::cout << pop.champion_x()[i] << "\n";
    }
  
  std::ofstream ofs("output/sunkcost_population.txt");
  boost::archive::text_oarchive oa(ofs);
  pop.save(oa,1);
  
  linebreak();

  time(&stop);
  printf("\nProgram complete! Total runtime: %0.16f seconds.\n",difftime(stop,start));


  return 0;
}
