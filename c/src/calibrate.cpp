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
#define ND 63 // number of destinations

extern "C" double Q;
extern "C" double kappa_x;
extern "C" double sig_z;
extern "C" double rho_z;
extern "C" double z_grid_mult_ub;
extern "C" double z_grid_mult_lb;
extern "C" double delta0;
extern "C" double delta1;
extern "C" double psi_n;
extern "C" double alpha_n;
extern "C" double beta_n;
extern "C" double gamma_n;
extern "C" double psi_o;
extern "C" double alpha_o;
extern "C" double beta_o;
extern "C" double gamma_o;
extern "C" int policy_solved_flag[ND];
extern "C" double L[ND];
extern "C" double La_n[ND];
extern "C" double La_o[ND];
extern "C" double Lam_n[ND];
extern "C" double Lam_o[ND];

extern "C" void linebreak();
extern "C" void linebreak2();
extern "C" int init_params();
extern "C" void discretize_x(int pareto);
extern "C" void discretize_z();
extern "C" void calc_survival_probs();
extern "C" void discretize_m();
extern "C" void random_draws();
extern "C" int solve_export_cost_all_dests();
extern "C" int solve_policies_all_dests();
extern "C" double simul_all_dests();
extern "C" void create_panel_dataset(const char * fname);

double fitness_func(double params[14])
{
  time_t start, stop;
  time(&start);

  fcnt++;
  
  kappa_x = params[0];
  sig_z = params[1];
  rho_z = params[2];
  psi_n = params[3];
  alpha_n = params[4];
  gamma_n = params[5];
  psi_o = params[6];
  alpha_o = params[7];
  gamma_o = params[8];
  Q = params[9];
  delta0 = params[10];
  delta1 = params[11];
  beta_n = params[12];
  beta_o = params[13];
  
  printf("\nCandidate parameter vector %d:\n",fcnt);
  printf("\tkappa_x =     %0.8f\n",kappa_x);
  printf("\tsig_z =       %0.8f\n",sig_z);
  printf("\trho_z =       %0.8f\n",rho_z);
  printf("\tpsi_n =       %0.8f\n",psi_n);
  printf("\talpha_n =     %0.8f\n",alpha_n);
  printf("\tbeta_n =     %0.8f\n",beta_n);
  printf("\tgamma_n =     %0.8f\n",gamma_n);
  printf("\tpsi_o =       %0.8f\n",psi_o);
  printf("\talpha_o =     %0.8f\n",alpha_o);
  printf("\tbeta_o =     %0.8f\n",beta_o);
  printf("\tgamma_o =     %0.8f\n",gamma_o);
  printf("\tQ =           %0.8f\n",Q);
  printf("\tdelta0 =       %0.8f\n",delta0);
  printf("\tdelta1 =       %0.8f\n\n",delta1);

  for(int id=0; id<ND; id++)
    {
      La_n[id] = pow(L[id],alpha_n);
      Lam_n[id] = pow(L[id],alpha_n-1.0);
      La_o[id] = pow(L[id],alpha_o);
      Lam_o[id] = pow(L[id],alpha_o-1.0);
    }
  
  discretize_x(0);
  discretize_z();
  calc_survival_probs();
  
  if(solve_export_cost_all_dests())
      return GSL_NAN;

  if(solve_policies_all_dests())
    return GSL_NAN;
    
  double expart_rate = simul_all_dests();
  create_panel_dataset("output/model_microdata_calibration.csv");


  printf("Calling python scripts to process and analyze simulated microdata\n");
  if(system("python3 -W ignore ../python/model_microdata_prep.py"))
    {
      printf("Failed to run script to process simulated microdata!\n");
      linebreak();
      return GSL_NAN;
    }
      
  if(system("python3 -W ignore ../python/sumstats_regs.py"))
    {
      printf("Failed to run script to compare data and model summary stats!\n");
      linebreak();
      return GSL_NAN;
    }

  if(system("python3 -W ignore ../python/life_cycle.py"))
    {
      printf("Failed to run script to compare data and model life cycle dynamics\n");
      linebreak();
      return GSL_NAN;
    }

  FILE * file = fopen("../python/output/calibration_data.txt","r");
  if(!file)
    {
      printf("Failed to open file with calibration data!\n");
      return GSL_NAN;
      linebreak();
    }
  else
    {
      double data_moments[14];
      double model_moments[14];
      double weights[14];

      int got = 0;
      for(int i=0; i<13; i++)
	{
	  got += fscanf(file,"%lf",&(data_moments[i]));
	}
      for(int i=0; i<13; i++)
	{
	  got += fscanf(file,"%lf",&(model_moments[i]));
	}
      for(int i=0; i<13; i++)
	{
	  got += fscanf(file,"%lf",&(weights[i]));
	}

      fclose(file);
      if(got!=3*13)
	{
	  printf("Failed to load calibration data!\n");
	  return GSL_NAN;
	  linebreak();
	}
      else
	{
	  data_moments[13] = 0.25;
	  model_moments[13] = expart_rate;
	  weights[13] = 0.25/10.0;

	  double error = 0.0;
	  double sum=0.0;
	  for(int i=0; i<14; i++)
	    {
	      weights[i]=1.0/weights[i];
	      double tmp = fabs(data_moments[i]-model_moments[i]);
	      sum += weights[i];
	      	      
	      error += weights[i]*tmp*tmp;
	    }

	  int fails = 0;
	  for (int id=0; id<ND; id++)
	    {
	      fails += policy_solved_flag[id];
	    }

	  //double tmp = (fails-ND);
	  //error += tmp*tmp;
	  error = error/sum;

	  time(&stop);

	  printf("\nMoments in data vs model:\n");
	  printf("\tTop 5 share (avg):             %0.4f %0.4f (%0.4f)\n",
		 data_moments[0],model_moments[0],weights[0]);
	  printf("\tTop 5 share (slope):           %0.4f %0.4f (%0.4f)\n",
		 data_moments[1],model_moments[1],weights[1]);
	  printf("\tAvg num dest (avg):            %0.4f %0.4f (%0.4f)\n",
		 data_moments[2],model_moments[2],weights[2]);
	  printf("\tAvg num dest (slope):         %0.4f %0.4f (%0.4f)\n",
		 data_moments[3],model_moments[3],weights[3]);
	  printf("\tExit rate (avg):               %0.4f %0.4f (%0.4f)\n",
		 data_moments[4],model_moments[4],weights[4]);
	  printf("\tExit rate (slope):            %0.4f %0.4f (%0.4f)\n",
		 data_moments[5],model_moments[5],weights[5]);
	  printf("\tEntrant rel size (avg):        %0.4f %0.4f (%0.4f)\n",
		 data_moments[6],model_moments[6],weights[6]);
	  printf("\tEntrant rel size (slope):     %0.4f %0.4f (%0.4f)\n",
		 data_moments[7],model_moments[7],weights[6]);
	  printf("\tEntrant rel exit rate (avg):   %0.4f %0.4f (%0.4f)\n",
		 data_moments[8],model_moments[8],weights[8]);
	  printf("\tEntrant rel exit rate (slope): %0.4f %0.4f (%0.4f)\n",
		 data_moments[9],model_moments[9],weights[9]);
	  printf("\t5-year sales increase (hard):  %0.4f %0.4f (%0.4f)\n",
		 data_moments[10],model_moments[10],weights[10]);
	  printf("\t5-year sales increase (easy):  %0.4f %0.4f (%0.4f)\n",
		 data_moments[11],model_moments[11],weights[11]);
	  printf("\t5-year decrease in exit rate: %0.4f %0.4f (%0.4f)\n",
		 data_moments[12],model_moments[12],weights[12]);
	  printf("\tExport participation rate:     %0.4f %0.4f (%0.4f)\n",
		 data_moments[13],model_moments[13],weights[13]);
	  printf("\tDestinations with errors:      %d\n",fails);
	  
	  printf("\nFitness evaluation complete! Runtime = %0.0f seconds. Error = %0.8f\n",difftime(stop,start),error);
	  
	  linebreak();

	  return error;
	}
    } 
}

using namespace pagmo;

// Objective function: utilitarian welfare criterion subject to box bounds
struct PAGMO_DLL_PUBLIC calibrate
  {
   // empty constuctor
   calibrate(){};

   // problem name
   std::string get_name() const
   {
     return "Export market penetration dymamics calibration objective function";
   }

   // Implementation of the objective function.
   vector_double fitness(const vector_double &dv) const
   {
     double params[14];
     for(int i=0; i<14; i++)
       {
	 params[i] = dv[i];
       }
     return {fitness_func(params)};
   }
  
   // Implementation of the box bounds.
   std::pair<vector_double, vector_double> get_bounds() const
   {
     /*
       kappa_x = params[0];
       sig_z = params[1];
       rho_z = params[2];
       psi_n = params[3];
       alpha_n = params[4];
       gamma_n = params[5];
       psi_o = params[6];
       alpha_o = params[7];
       gamma_o = params[8];
       Q = params[9];
       delta0 = params[10];
       delta1 = params[11];
       delta0 = params[12];
       delta1 = params[13];
     */
     return {{0.7, 0.3, 0.6,
	      0.08, 0.3, 4.5,
	      0.08, 0.5, 1.35,
	      0.6, 6.0, 0.0025, 0.25, 0.6},
	     {0.9, 0.4, 0.85,
	      0.15, 0.8, 7.0,
	      0.15, 0.95, 3.5,
	      0.9, 30., 0.025, 1.0, 1.0}};
   }

   template <typename Archive>
   void serialize(Archive &ar, unsigned){}
  };

PAGMO_S11N_PROBLEM_EXPORT_KEY(calibrate)
PAGMO_S11N_PROBLEM_IMPLEMENT(calibrate)

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
  discretize_m();
  random_draws();
  
  fcnt=0;
  
  problem prob{calibrate{}};
  std::cout << "\n" << prob << '\n';
  
  algorithm algo{de1220(20)};
  algo.set_verbosity(1);
  std::cout << "\n" << algo << '\n';
  
  linebreak();
  
  population pop{prob, 60};
  //population pop{problem{}};
  //std::ifstream ifs("output/calibration_population.txt");
  //boost::archive::text_iarchive ia(ifs);
  //pop.load(ia,1);

  //std::cout << "\n" << pop << '\n';
  //linebreak();
    
  //pop = algo.evolve(pop);

  linebreak();
  std::cout << "\nChampion:\n";
  for(int i=0; i<14; i++)
    {
      std::cout << pop.champion_x()[i] << "\n";
    }

  //std::ofstream ofs("output/calibration_population.txt");
  //boost::archive::text_oarchive oa(ofs);
  //pop.save(oa,1);

  linebreak();

  nlopt local("sbplx");
  local.set_selection("best");
  local.set_replacement("best");
  local.set_maxeval(500);
  local.set_verbosity(1);
  //std::cout << "\n" << local << '\n';

  linebreak();
  
  pop = local.evolve(pop);

  linebreak();
  std::cout << "\nNew champion:\n";
  for(int i=0; i<14; i++)
    {
      std::cout << pop.champion_x()[i] << "\n";
    }
  
  std::ofstream ofs("output/calibration_population.txt");
  boost::archive::text_oarchive oa(ofs);
  pop.save(oa,1);
  
  linebreak();

  time(&stop);
  printf("\nProgram complete! Total runtime: %0.16f seconds.\n",difftime(stop,start));


  return 0;
}
