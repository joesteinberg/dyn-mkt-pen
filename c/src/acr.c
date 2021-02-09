///////////////////////////////////////////////////////////////////////////////
// FIXED_COST.C
// Joseph B. Steinberg, University of Toronto
//
// This program performs analyses using the sunk-cost Alessandria and Choi model
// in the paper "Export Market Penetration Dynamics."
// The code is organized into sections:
// 
//	1. Includes, macros, and computational utilities
//
//	2. Parameters (including destination data) and inline functions
//
//      3. Static export cost function f(m,m')
//
//	4. Iteration procedure to solve for dynamic policy functions
//
//	5. Simulation of panel of exporters
//
//	6. Deterministic aggregate transition dynamics
//
//	7. Life cycles
//
//	8. Main function
//
// To compile the program, the user must have the following libraries:
// 	i. GNU GSL library (I have used version 2.1)
//	ii. OpenMP
// There is a makefile in the same directory as the source code. To compile
// and run the program, simply type "make model" followed by "./bin/sunk_cost"


///////////////////////////////////////////////////////////////////////////////
// 1. Includes, macros, etc.
///////////////////////////////////////////////////////////////////////////////

// includes
#include <unistd.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_types.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_linalg.h>
#include <omp.h>

// macros: discretization
#define NX 50 // fixed-effect grid size
#define NZ 101 // productivity shock grid size
#define ND 63 // number of destinations
#define NT 100 // simulation length
#define NF 50000 // simulation population size

// macros: paralellization
#ifdef _OPENMP
#define PAR 1
#else
#define PAR 0
#endif

// macros: tolerances
const int root_max_iter = 100;
const double root_tol_rel = 1.0e-6;
const double root_tol_abs = 1.0e-6;
//const int min_max_iter = 100;
const double delta_deriv = 1.0e-9;
//const double delta_min = 0.00001;
const double delta_root = 1.0e-9;
const double policy_tol_abs = 1.0e-8;
const int policy_max_iter = 2000;
const double x_grid_ub_mult = 10.0;
const double x_grid_exp = 1.0;

// print verbose output y/n
const int verbose=1;

// initialize all elements of an array to the same numeric value
void set_all_v(double * v, int n, double val)
{
  int i;
  for(i=0; i<n; i++)
    {
      v[i]=val;
    }
}

#define SET_ALL_V(v,n,val) ( set_all_v( (double *)(v), (n), (val) ) )

// sum squared error
double enorm(const gsl_vector * f) 
{
  double e2 = 0 ;
  size_t i, n = f->size ;
  for (i = 0; i < n ; i++) {
    double fi= gsl_vector_get(f, i);
    e2 += fi * fi ;
  }
  return sqrt(e2);
}

// sum elements of vector
double vsum(const gsl_vector * f)
{
  double sum = 0 ;
  size_t i, n = f->size ;
  for (i = 0; i < n ; i++) {
    double fi= gsl_vector_get(f, i);
    sum += fabs(fi) ;
  }
  return sum;
}

// write array to text file
void write_vector(const double * v, int n, char * fname)
{
  FILE * file = fopen(fname,"wb");
  int i;
  for(i=0; i<n; i++)
    {
      fprintf(file,"%0.16f\n",v[i]);
    }
  fclose(file);
}

// macro used for writing slices of multidimensional statically allocated arrays
#define WRITE_VECTOR(v,n,fname) ( write_vector( (double *)(v), (n), (fname) ) )

// pareto distribution cdf
double pareto_cdf(double x, double kappa)
{
  return 1.0 - pow(x,-kappa);
}

// pareto distribution pdf
double pareto_pdf(double x, double kappa)
{
  return kappa*pow(x,-kappa-1.0);
}

// pareto distribution inverse cdf
double pareto_cdf_inv(double P, double kappa)
{
  return pow(1.0-P,-1.0/kappa);
}

// linspace
void linspace(double lo, double hi, int n, double * v)
{
  double d=(hi-lo)/(n-1.0);
  v[0]=lo;
  int i=0;
  for(i=1;i<n;i++)
    {
      v[i] = v[i-1]+d;
    }
}

// expspace
void expspace(double lo, double hi, int n, double ex, double * v)
{
  linspace(0.0,pow(hi-lo,1.0/ex),n,v);
  int i;
  for(i=0;i<n;i++)
    {
      v[i] = pow(v[i],ex)+lo;
    }
  return;
}

// linear interpolation
static inline double interp(gsl_interp_accel * acc, const double *xa, const double *ya, int n, double x)
{
  double x0=0.0;
  double x1=0.0;
  double xd=0.0;
  double q0=0.0;
  double q1=0.0;
  double retval=0.0;

  int ix = gsl_interp_accel_find(acc, xa, n, x);

  if(ix==0)
    {
      x0 = xa[0];
      x1 = xa[1];
      xd = x1-x0;
      q0 = ya[0];
      q1 = ya[1];
    }
  else if(ix==n-1)
    {
      x0 = xa[n-2];
      x1 = xa[n-1];
      xd = x1-x0;
      q0 = ya[n-2];
      q1 = ya[n-1];
    }
  else
    {
      x0 = xa[ix];
      x1 = xa[ix+1];
      xd = x1-x0;
      q0 = ya[ix];
      q1 = ya[ix+1];
    }

  retval = ( q0*(x1-x) + q1*(x-x0) ) / xd;
  return retval;
}

// linear interpolation
static inline double interp_with_ix(const double *xa, const double *ya, int n, double x, int ix)
{
  double x0=0.0;
  double x1=0.0;
  double xd=0.0;
  double q0=0.0;
  double q1=0.0;
  double retval=0.0;

  if(ix==0)
    {
      x0 = xa[0];
      x1 = xa[1];
      xd = x1-x0;
      q0 = ya[0];
      q1 = ya[1];
    }
  else if(ix==n-1)
    {
      x0 = xa[n-2];
      x1 = xa[n-1];
      xd = x1-x0;
      q0 = ya[n-2];
      q1 = ya[n-1];
    }
  else
    {
      x0 = xa[ix];
      x1 = xa[ix+1];
      xd = x1-x0;
      q0 = ya[ix];
      q1 = ya[ix+1];
    }

  retval = ( q0*(x1-x) + q1*(x-x0) ) / xd;
  return retval;
}

void markov_stationary_dist(int n, double P[n][n], double p[n])
{
  SET_ALL_V(p,n,1.0/n);
  double diff=-HUGE_VAL;
  do
    {
      diff=-HUGE_VAL;
      
      double tmp[n];
      
      for(int i=0; i<n; i++)
	{
	  tmp[i]=0.0;
	  
	  for(int j=0; j<n; j++)
	    {
	      tmp[i] += P[j][i]*p[j];
	    }
	}
      for(int i=0; i<n; i++)
	{
	  if(fabs(tmp[i]-p[i])>diff)
	    {
	      diff=fabs(tmp[i]-p[i]);
	    }
	}
       for(int i=0; i<n; i++)
	 {
	   p[i]=tmp[i];
	 }
    }
  while(diff>1.0e-11);
}

// root finder
int find_root_1d(gsl_function * f, double xlo, double xhi, double * x)
{
  int status = 0;
  int iter = 0;
  const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
  gsl_root_fsolver * s = gsl_root_fsolver_alloc(T);
  
  status = gsl_root_fsolver_set(s,f,xlo,xhi);
  if(status)
    {
      printf("Error initializing root-finder!\n");
    }
  else
    {
      do
	{
	  iter++;
	  status = gsl_root_fsolver_iterate(s);
	  if(status)
	    {
	      printf("Error iterating root-finder!\n");
	      break;
	    }
	  *x = gsl_root_fsolver_root(s);
	  xlo = gsl_root_fsolver_x_lower(s);
	  xhi = gsl_root_fsolver_x_upper(s);
	  status = gsl_root_test_interval(xlo,xhi,root_tol_abs,root_tol_rel);
	}while(status==GSL_CONTINUE && iter<root_max_iter);
    }

  gsl_root_fsolver_free(s);

  return status;
  
}

void linebreak()
{
  printf("\n////////////////////////////////////////////////////////////////////////////\n\n");
}

void linebreak2()
{ 
  printf("\n----------------------------------------------------------------------------\n");
}

///////////////////////////////////////////////////////////////////////////////
// 2. Declarations of parameters, grids, and inline functions
///////////////////////////////////////////////////////////////////////////////

int alpha_flag=0;

// parameters
double W = 0.0; // wage (note: represents normalization of export country GDP per capita relative to representative destination)
double Q = 0.0; // discount factor
double delta0 = 0.0; // survival rate
double delta1 = 0.0; // survival rate
double theta = 0.0; // EoS between varieties
double theta_hat = 0.0; // = (1/theta)*(theta/(theta-1))^(1-theta)
double kappa_x = 0.0; // fixed productivity tail parameter
double sig_z = 0.0; // stochastic productivity dispersion
double rho_z = 0.0; // stochastic productivity persistence
double corr_z = 0.0; // correlation of productivity shock innovations across destinations
double kappa0; // entry cost
double kappa1; // continuation cost
double alpha0;
double alpha1;
double xi; // iceberg cost in high state
double rho0; // transition probability from low state
double rho1; // transition probability from high state 

double z_grid_mult_lb = 0.0;
double z_grid_mult_ub = 0.0;

// fixed effect grid
double x_grid[NX] = {0.0}; // grid
double x_hat[NX] = {0.0}; // x^{theta-1} grid
double x_probs[NX] = {0.0}; // probabilities
double x_cumprobs[NX] = {0.0}; // cumultative probabilities
double delta[NX][NZ] = {{0.0}};

// productivity shock grid
double z_grid[NZ] = {0.0}; // grid
double z_hat[NZ] = {0.0}; // z^{theta-1} grid
double z_ucond_probs[NZ] = {0.0}; // ergodic probabilities
double z_ucond_cumprobs[NZ] = {0.0}; // cumulative ergodic probabilities
double z_trans_probs[NZ][NZ] = {{0.0}}; // transition probabilities
double z_trans_cumprobs[NZ][NZ] = {{0.0}}; // cumultative transition probabilities

// expected productivity tomorrow
double E_xhat_zhat[NX][NZ] = {{0.0}};

// destination-specific parameters
char name[ND][3] = {{""}}; // name
double L[ND] = {0.0}; // market size
double Y[ND] = {0.0}; // aggregate consumption index
double L_a0[ND] = {0.0};
double L_a1[ND] = {0.0};
//double tau[ND] = {0.0}; // trade cost
double tau_hat[ND] = {0.0}; // = tau^(1-theta)
double pi_hat[ND] = {0.0}; // theta_hat*L*Y*tau_hat

void discretize_x(int pareto)
{
  if(pareto)
    {
      double x_lo=1.0;
      double x_hi=x_grid_ub_mult*kappa_x;
      expspace(x_lo,x_hi,NX,x_grid_exp,x_grid);

      double sum = 0.0;
      for(int i=1; i<NX; i++)
	{
	  x_probs[i] = pareto_cdf(x_grid[i],kappa_x)-pareto_cdf(x_grid[i-1],kappa_x);
	  x_cumprobs[i] = x_probs[i] +sum;
	  sum += x_probs[i];
	}
      x_probs[0] = 1.0 - sum;
    }
  else
    {
      double sum=0.0;
      double m[NX-1];
      for(int i=0; i<NX; i++)
	{
	  if(i<NX-1)
	    m[i] = gsl_cdf_ugaussian_Pinv( ((double)(i+1))/((double)(NX)) ) * kappa_x;
	  
	  x_probs[i] = 1.0/NX;
	  sum += x_probs[i];

	  if(i==0)
	    x_cumprobs[i] = x_probs[i];
	  else
	    x_cumprobs[i] = x_cumprobs[i-1] + x_probs[i];
	}

      if(fabs(sum-1.0)>1.0e-8)
	printf("X probs dont sum to 1!! %0.8f\n",sum);

      x_grid[0] = exp(-kappa_x*NX*gsl_ran_gaussian_pdf(m[0]/kappa_x,1.0));
      for(int i=1; i<(NX-1); i++)
	{
	  x_grid[i] = exp(-kappa_x*NX*(gsl_ran_gaussian_pdf(m[i]/kappa_x,1.0)-gsl_ran_gaussian_pdf(m[i-1]/kappa_x,1.0)));
	}
      x_grid[NX-1] = exp(kappa_x*NX*gsl_ran_gaussian_pdf(m[NX-2]/kappa_x,1.0));
    }
  
  for(int i=0; i<NX; i++)
    {
      x_hat[i] = pow(x_grid[i],theta-1.0);
    }
  

  return;
}

void discretize_z()
{
  int n = NZ;
  int i,j;
  double mup = z_grid_mult_ub;
  double mdown = z_grid_mult_lb;
  double ucond_std = sqrt(sig_z*sig_z/(1.0-rho_z*rho_z));
  double lo = -mdown*ucond_std;
  double hi = mup*ucond_std;
  double d = (hi-lo)/(n-1.0);
  linspace(lo,hi,n,z_grid);
 
  for(i=0; i<n; i++)
    {
      double x = z_grid[i];
	
      for(j=0; j<n; j++)
	{
	  double y = z_grid[j];
	  
	  if(j==0)
	    {
	      z_trans_probs[i][j] = gsl_cdf_ugaussian_P( (y + d/2.0 - rho_z*x) / sig_z);
	    }
	  else if(j==(n-1))
	    {
	      z_trans_probs[i][j] = 1.0 - gsl_cdf_ugaussian_P( (y - d/2.0 - rho_z*x) / sig_z);
	    }
	  else
	    {
	      z_trans_probs[i][j] = (gsl_cdf_ugaussian_P( (y + d/2.0 - rho_z*x) / sig_z) -
			  gsl_cdf_ugaussian_P( (y - d/2.0 - rho_z*x) / sig_z));
	    }
	}
    }  

  markov_stationary_dist(NZ, z_trans_probs, z_ucond_probs);
  
  double sum=0.0;
  for(i=0; i<n; i++)
    {
      z_grid[i] = exp(z_grid[i]);
      z_hat[i] = pow(z_grid[i],theta-1.0);
      z_ucond_cumprobs[i] = z_ucond_probs[i] + sum;
      sum += z_ucond_probs[i];
    }
  if(fabs(sum-1.0)>1.0e-10)
    {
      printf("warning: ergodic probabilities do not sum to 1 for state %d!\n",i);
    }

  
  for(i=0; i<n; i++)
    {
      sum = 0.0;
      
      for(j=0; j<n; j++)
	{
	  z_trans_cumprobs[i][j] = z_trans_probs[i][j] + sum;
	  sum += z_trans_probs[i][j];
	}
      if(fabs(sum-1.0)>1.0e-10)
	{
	  printf("warning: transition probabilities do not sum to 1 for state %d!\n",i);
	}
    }
}

void calc_expected_productivity()
{
  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  double death_prob=fmax(0.0,fmin(exp(-delta0*x_hat[ix]*z_hat[iz])+delta1,1.0));
	  //double death_prob=fmax(0.0,fmin(exp(-delta0*x_hat[ix])+delta1,1.0));
	  delta[ix][iz] = 1.0-death_prob;
	  
	  E_xhat_zhat[ix][iz] = 0.0;
	  for(int izp=0; izp<NZ; izp++)
	    {
	      E_xhat_zhat[ix][iz] += z_trans_probs[iz][izp]*x_hat[ix]*z_hat[izp];
	    }
	}
    }
}

// assigned parameters and initial guesses
int init_params()
{  
  // initial guesses!!!
  W = 1.0;
  Q = 0.86245704;
  delta0 = 28.95796345;
  delta1 = 0.01300775;
  theta = 5.0;
  theta_hat = (1.0/theta) * pow(theta/(theta-1.0),1.0-theta);
  kappa_x = 0.8576352;
  sig_z = 0.35;
  rho_z =  0.75;

  if(alpha_flag==0)
    {
      kappa0 = 1.1;
      kappa1 = 1.4;
      alpha0=0.0;
      alpha1=0.0;
      xi=1.7/1.07;
      rho0=0.9;
      rho1=0.9;
    }
  else if(alpha_flag==1)
    {
      kappa0 = 5.0;
      kappa1 = 1.5;
      alpha0 = 0.5459469;
      alpha1 = 0.8409127;
      xi=1.72/1.07;
      rho0=0.9;
      rho1=0.9;
    }

  z_grid_mult_lb=3.0;
  z_grid_mult_ub=3.0;

  // set all destination-specific variables to mean values... we will use the
  // array of destinations in parallelizing the calibration
  FILE * file = fopen("../python/output/dests_for_c_program.txt","r");
  if(!file)
    {
      printf("Failed to open file with destination data!\n");
      return 1;
    }
  else
    {
      char buffer[3];
      double pop, gdppc, tau_;
      int got;

      int id;
      for(id=0; id<ND; id++)
	{
	  got = fscanf(file,"%s %lf %lf %lf",buffer,&pop,&gdppc,&tau_);
	  if(got!=4)
	    {
	      printf("Failed to load data for destination %d!\n",id);
	      fclose(file);
	      return 1;
	    }
	  else
	    {
	      L[id] = pop;
	      L_a0[id] = pow(L[id],alpha0);
	      L_a1[id] = pow(L[id],alpha1);
	      Y[id] = gdppc;
	      tau_hat[id] = 1.0/tau_;
	      strncpy(name[id],buffer,3);
	      //tau_hat[id] = pow(tau[id],1.0-theta);
	      pi_hat[id] = theta_hat * L[id] * Y[id] * tau_hat[id];
	    }
	}

      return 0;
    }
}


///////////////////////////////////////////////////////////////////////////////
// 3. Dynamic program
///////////////////////////////////////////////////////////////////////////////

double V[ND][NZ][NZ][3] = {{{{0.0}}}};
double EV[ND][NZ][NZ][3] = {{{{0.0}}}};
int gex[ND][NZ][NZ][3] = {{{{0}}}};
double expart_rate[ND] = {0.0};
int policy_solved_flag[ND] = {0};

void init_dp_objs(int id)
{
  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  V[id][ix][iz][0] = 0.0;
	  V[id][ix][iz][1] = pi_hat[id]*pow(xi,1.0-theta)*x_hat[ix]*z_hat[iz]/Q;
	  V[id][ix][iz][2] = pi_hat[id]*x_hat[ix]*z_hat[iz]/Q;
	}
    }
}

void calc_EV(int id)
{
  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  for(int is=0; is<3; is++)
	    {
	      EV[id][ix][iz][is]=0.0;
	      for(int izp=0; izp<NZ; izp++)
		{
		  if(z_trans_probs[iz][izp]>1.0e-11)
		    {
		      EV[id][ix][iz][is] += Q*delta[ix][iz]*V[id][ix][izp][is]*z_trans_probs[iz][izp];
		    }
		}
	    }		     
	}
    }
  
}

void iterate_policies(int id, double * maxdiff, int imaxdiff[3])
{
  *maxdiff=-HUGE_VAL;

  calc_EV(id);
  
  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  double pi = pi_hat[id]*x_hat[ix]*z_hat[iz];
	  
	  if(EV[id][ix][iz][0]<
	     pi*pow(xi,1.0-theta) + EV[id][ix][iz][1] - kappa0*L_a0[id])
	    {
	      gex[id][ix][iz][0] = 1;
	    }
	  else
	    {
	      gex[id][ix][iz][0] = 0;
	    }

	  if(EV[id][ix][iz][0]<
	     pi*pow(xi,1.0-theta) - kappa1*L_a1[id] +
	     rho0*EV[id][ix][iz][1] + (1.0-rho0)*EV[id][ix][iz][2])
	    {
	      gex[id][ix][iz][1] = 1;
	    }
	  else
	    {
	      gex[id][ix][iz][1] = 0;
	    }
	  
	  if(EV[id][ix][iz][0]<
	     pi - kappa1*L_a1[id] +
	     (1.0-rho1)*EV[id][ix][iz][1] + rho1*EV[id][ix][iz][2])
	    {
	      gex[id][ix][iz][2] = 1;
	    }
	  else
	    {
	      gex[id][ix][iz][2] = 0;
	    }
	  
	  double tmp0 = fmax(EV[id][ix][iz][0],
			     pi*pow(xi,1.0-theta) - kappa0*L_a0[id] +
			     EV[id][ix][iz][1]);
	  
	  double tmp1 = fmax(EV[id][ix][iz][0],
			     pi*pow(xi,1.0-theta) - kappa1*L_a1[id] +
			     rho0*EV[id][ix][iz][1] +
			     (1.0-rho0)*EV[id][ix][iz][2]);

	  double tmp2 = fmax(EV[id][ix][iz][0],
			     pi - kappa1*L_a1[id] +
			     (1.0-rho1)*EV[id][ix][iz][1] +
			     rho1*EV[id][ix][iz][2]);

	  
	  double diff0 = fabs(tmp0-V[id][ix][iz][0]);
	  double diff1 = fabs(tmp1-V[id][ix][iz][1]);
	  double diff2 = fabs(tmp2-V[id][ix][iz][2]);

	  if(diff0>*maxdiff)
	    {
	      *maxdiff=diff0;
	      imaxdiff[0]=ix;
	      imaxdiff[1]=iz;
	      imaxdiff[2]=0;
	    }

	  if(diff1>*maxdiff)
	    {
	      *maxdiff=diff1;
	      imaxdiff[0]=ix;
	      imaxdiff[1]=iz;
	      imaxdiff[2]=1;
	    }
	  
	  if(diff2>*maxdiff)
	    {
	      *maxdiff=diff2;
	      imaxdiff[0]=ix;
	      imaxdiff[1]=iz;
	      imaxdiff[2]=2;
	    }

	  
	  V[id][ix][iz][0] = tmp0;
	  V[id][ix][iz][1] = tmp1;
	  V[id][ix][iz][2] = tmp2;
	}
    }  
}

int solve_policies(int id)
{
  time_t start, stop;
  time(&start);

  init_dp_objs(id);

  int status = 0;
  double maxdiff = 999;
  int imaxdiff[3];
  
  int iter=0;
  do
    {
      iter++;
      iterate_policies(id,&maxdiff,imaxdiff);

      if(verbose==3)
	printf("\t\tIter %d, diff = %0.2g, loc=(%d,%d,%d), ex=%d, V=%0.4g\n",
	       iter,maxdiff,imaxdiff[0],imaxdiff[1],imaxdiff[2],
	       gex[id][imaxdiff[0]][imaxdiff[1]][imaxdiff[2]],
	       V[id][imaxdiff[0]][imaxdiff[1]][imaxdiff[2]]);
    }
  while(maxdiff>policy_tol_abs && iter < policy_max_iter);

  expart_rate[id] = 0.0;
  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  expart_rate[id] += x_probs[ix]*z_ucond_probs[iz]*gex[id][ix][iz][0];
	}
    }    

  time(&stop);

  if(iter==policy_max_iter)
    {
      status=1;
      if(verbose==2)
	printf("\tValue function iteration failed for %.3s! Diff = %0.4g\n",
	       name[id],maxdiff);
    }
  else
    {
      if(verbose==2)
	{
	  printf("\tValue function converged for %.3s in %0.0f seconds!",
		 name[id],difftime(stop,start));
	  printf(" Export participation rate = %0.8f.\n",
		 100*expart_rate[id]);
	}
    }

  return status;

}

int solve_policies_all_dests()
{
  if(verbose)
    printf("\nSolving dynamic programs for all destinations...\n");

  time_t start, stop;
  time(&start);

  int cnt=0;
  
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int id=0; id<ND; id++)
    {
      policy_solved_flag[id] = solve_policies(id);
      cnt += policy_solved_flag[id];
    }

  time(&stop);
  
  if(verbose)
    {
      printf("Finished dynamic programs in %0.0f seconds. %d failed to converge.\n",difftime(stop,start),cnt);
    }
  
  return 0;  
}

///////////////////////////////////////////////////////////////////////////////
// 4. Simulation
///////////////////////////////////////////////////////////////////////////////

// storage for simulated data
// we use 3*NT to store NT throwaway periods, NT periods to simulate for calibration,
// and NT periods for the shock analysis
unsigned long int seed = 0;
double x_rand[NF];
double z_rand[ND][NF][NT*2];
double switch_rand[ND][NF][NT*2];
double surv_rand[NF][NT*2];
int ix_sim[NF];
int iz_sim[ND][NF][NT*2];
double v_sim[ND][NF][NT*2];

// draw random variables
void random_draws()
{
  printf("\nDrawing random numbers for simulation...\n");
  
  time_t start, stop;
  time(&start);
  
  gsl_rng_env_setup();
  gsl_rng * r = gsl_rng_alloc(gsl_rng_default);

  for(int id=0; id<ND; id++)
    {
      for(int i=0; i<NF; i++)
	{
	  if(id==0)
	    x_rand[i] = gsl_rng_uniform(r);
	  
	  for(int t=0; t<NT*2; t++)
	    {
	      z_rand[id][i][t] = gsl_rng_uniform(r);
	      switch_rand[id][i][t] = gsl_rng_uniform(r);
	      if(id==0)
		surv_rand[i][t] = gsl_rng_uniform(r);
	    }
	}
    }

  gsl_rng_free(r);

  time(&stop);
  printf("Random draws finished! Time = %0.0f\n",difftime(stop,start));
}

// main simulation function
void simul(int id)
{
  //if(verbose)
  //  printf("\n\tSimulating model for id=%d...\n",id);

  time_t start, stop;
  time(&start);

  int max_kt = NT*2;

  gsl_interp_accel * acc1 = gsl_interp_accel_alloc();
  gsl_interp_accel * acc2 = gsl_interp_accel_alloc();

  // then for each firm in the sample...
  for(int jf=0; jf<NF; jf++)
    {
      // find fixed-effect value based on random draw
      gsl_interp_accel_reset(acc1);
      int ix = gsl_interp_accel_find(acc1, x_cumprobs, NX, x_rand[jf]);
      if(id==0)
	ix_sim[jf] = ix;

      if(ix<0 || ix>=NX)
	{
	  printf("Error!\n");
	}
      
      // find initial value of shock based on random draw and ergodic distribution
      gsl_interp_accel_reset(acc1);
      int iz = gsl_interp_accel_find(acc1, z_ucond_cumprobs, NZ, z_rand[id][jf][0]);
      iz_sim[id][jf][0] = iz;
      
      // start off as a non-exporter
      int s=0;
      
      for(int kt=0; kt<max_kt; kt++)
	{
	  if(surv_rand[jf][kt]>delta[ix][iz])
	    {
	      v_sim[id][jf][kt] = -99.9;
	      s=0;
	      
	      if(iz<max_kt-1)
		iz_sim[id][jf][kt+1] = gsl_interp_accel_find(acc1, z_ucond_cumprobs, NZ, z_rand[id][jf][kt+1]);
	    }
	  else
	    {
	      if(gex[id][ix][iz][s])
		{
		  if(s==0)
		    {
		      s=1;
		      v_sim[id][jf][kt] = theta*theta_hat*L[id]*Y[id]*tau_hat[id]*x_hat[ix]*z_hat[iz]*pow(xi,1.0-theta);
		    }
		  else if (s==1)
		    {
		      v_sim[id][jf][kt] = theta*theta_hat*L[id]*Y[id]*tau_hat[id]*x_hat[ix]*z_hat[iz]*pow(xi,1.0-theta);

		      if(switch_rand[id][jf][kt+1]<rho0)
			{
			  s=1;
			}
		      else
			{
			  s=2;
			}
		    }
		  else if(s==2)
		    {
		      v_sim[id][jf][kt] = theta*theta_hat*L[id]*Y[id]*tau_hat[id]*x_hat[ix]*z_hat[iz];

		      if(switch_rand[id][jf][kt+1]<rho1)
			{
			  s=2;
			}
		      else
			{
			  s=1;
			}
		    }
		}
	      else
		{
		  s=0;
		  v_sim[id][jf][kt] = -99.9;
		}

	      if(kt<max_kt-1)
		iz_sim[id][jf][kt+1] = gsl_interp_accel_find(acc1, z_trans_cumprobs[iz], NZ, z_rand[id][jf][kt+1]);

	    }

	  if(kt<max_kt-1)
	    iz = iz_sim[id][jf][kt+1];
	}      
    }

  double z_mass[NZ] = {0.0};
  double x_mass[NX] = {0.0};
  double expart_rate_[NT] = {0.0};
  double avg_expart_rate=0.0;
  for(int kt=NT; kt<NT*2; kt++)
    {
      for(int jf=0; jf<NF; jf++)
	{
	  z_mass[iz_sim[id][jf][kt]] += 1.0;
	  x_mass[ix_sim[jf]] += 1.0;
	  if(v_sim[id][jf][kt]>1.0e-10)
	    {
	      expart_rate_[kt-NT] += 1.0;
	    }
	}
      expart_rate_[kt-NT] = expart_rate_[kt-NT]/NF;
      avg_expart_rate += expart_rate_[kt-NT];
    }

  avg_expart_rate=avg_expart_rate/NT;

  gsl_interp_accel_free(acc1);
  gsl_interp_accel_free(acc2);

  time(&stop);

  if(verbose==2)
    printf("\tSimulation completed for %.3s in %0.0f seconds. Export part rate = %0.8f.\n",name[id],difftime(stop,start),100*avg_expart_rate);

  return;
}

void simul_all_dests(double *avg_expart_rate,
		     double *avg_exit_rate,
		     double *avg_nd,
		     double *new_size)
{
  if(verbose)
    printf("\nSimulating for all destinations...\n");

  time_t start, stop;
  time(&start);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int id=0; id<ND; id++)
    {
      if(policy_solved_flag[id]==0)
	{
	  simul(id);
	}
    }

  double expart_rate_[NT] = {0.0};
  *avg_expart_rate=0.0;
  int min_kt=NT;
  int max_kt=NT*2;
  for(int kt=min_kt; kt<max_kt; kt++)
    {
      for(int jf=0; jf<NF; jf++)
	{
	  int exporter=0;
	  for(int id=0; id<ND; id++)
	    {	      
	      if(policy_solved_flag[id]==0 && expart_rate[id]>1.0e-10 &&
		 exporter==0 && v_sim[id][jf][kt]>1.0e-10)
		{
		  expart_rate_[kt-min_kt] += 1.0;
		  exporter=1;
		}
	    }
	}
      expart_rate_[kt-min_kt] = expart_rate_[kt-min_kt]/NF;
      *avg_expart_rate += expart_rate_[kt-min_kt];
    }

  *avg_expart_rate = *avg_expart_rate/(max_kt-min_kt);

  //double exit_rate_[ND] = {{0.0}};
  double exit_rate_[ND] = {0.0};
  double avg_nd_[ND] = {0.0};
  double new_size_[ND] = {0.0};
  int ND2=0;
  int ND3=0;
  for(int id=0; id<ND; id++)
    {
      if(policy_solved_flag[id]==0 && expart_rate[id]>1.0e-10)
	{
	  ND2++;
	  ND3++;
	  int NT2=0;
	  int NT3=0;
	  for(int kt=min_kt; kt<max_kt; kt++)
	    {
	      int numex=0;
	      int exits=0;
	      int avg_nd2=0;
	      int num_new=0;
	      int num_inc=0;
	      double new_size2=0.0;
	      double inc_size2=0.0;
	      for(int jf=0; jf<NF; jf++)
		{
		  if(v_sim[id][jf][kt]>1.0e-10)
		    {
		      numex++;

		      int nd=0;
		      for(int id2=0; id2<ND; id2++)
			{
			  if(v_sim[id2][jf][kt]>1.0e-10)
			    nd++;
			}
		      avg_nd2 += nd;
		    }
	      
		  if(v_sim[id][jf][kt-1]>1.0e-10 && v_sim[id][jf][kt]<0.0)
		    {
		      exits ++;
		    }

		  if(v_sim[id][jf][kt-1]<0.0 && v_sim[id][jf][kt]>1.0e-10)
		    {
		      num_new++;
		      new_size2 += v_sim[id][jf][kt];
		    }
		  else if(v_sim[id][jf][kt-1]>1.0e-10 &&
			  v_sim[id][jf][kt]>1.0e-10)
		    {
		      num_inc++;
		      inc_size2 += v_sim[id][jf][kt];
		    }
		}

	      if(numex>0)
		{
		  NT2++;
		  exit_rate_[id] += ((double)(exits))/((double)(numex));
		  avg_nd_[id] += ((double)(avg_nd2))/((double)(numex));

		  if(num_new>0 && num_inc>0)
		    {
		      NT3++;
		      
		      new_size_[id] += (new_size2/((double)(num_new))) /
			(inc_size2/((double)(num_inc)));
		    }
		}
	    }
	  if(NT2>0)
	    {
	      exit_rate_[id]= exit_rate_[id]/NT2;
	      avg_nd_[id]= avg_nd_[id]/NT2;
	    }
	  else
	    {
	      exit_rate_[id]= 0.0;
	      avg_nd_[id]= 0.0;
	      ND2--;
	      expart_rate[id]=0.0;
	      new_size_[id]=0.0;
	    }

	  if(NT3>0)
	    {
	      new_size_[id] = new_size_[id]/NT3;
	    }
	  else
	    {
	      new_size_[id] = -999;
	      ND3--;
	    }
	}
    }

  *avg_exit_rate = 0.0;
  *avg_nd = 0.0;
  *new_size=0.0;
  for(int id=0; id<ND; id++)
    {
      if(policy_solved_flag[id]==0 && expart_rate[id]>1.0e-10)
	{
	  *avg_exit_rate += exit_rate_[id];
	  *avg_nd += avg_nd_[id];

	  if(new_size_[id]>1.0e-6)
	    *new_size += new_size_[id];
	}
    }
  *avg_exit_rate = *avg_exit_rate/((double)(ND2));
  *avg_nd = *avg_nd/((double)(ND2));
  *new_size = *new_size/((double)(ND3));
	
  time(&stop);

  if(verbose)
    printf("Finished simulations in %0.0f seconds.\n\tOverall export part. rate = %0.8f\n\tavg. exit rate = %0.8f\n\tavg. num. dests = %0.8f\n\tavg. new exporter relative size = %0.8f",
	   difftime(stop,start),100*(*avg_expart_rate),
	   100*(*avg_exit_rate),*avg_nd,*new_size);
    
  return;
}


void create_panel_dataset(const char * fname)
{
  if(verbose)
    printf("\nCreating panel dataset from simulation...\n");

  time_t start, stop;
  time(&start);

  int min_kt = NT;
  int max_kt = NT*2;
  
  FILE * file = fopen(fname,"w");

  int max_nd=0;
  
  fprintf(file,"f,d,y,popt,gdppc,tau,v,ix,iz,nd,nd_group\n");
  for(int jf=0; jf<NF; jf++)
    {
      for(int kt=min_kt; kt<max_kt; kt++)
	{
	  int nd=0;
	  for(int id=0; id<ND; id++)
	    {
	      if(policy_solved_flag[id]==0 && v_sim[id][jf][kt]>1.0e-10)
		{
		  nd++;
		}
	    }	  
	  if(nd>max_nd)
	    max_nd=nd;
	  
	  int nd_group=0;
	  if(nd<=4)
	    {
	      nd_group=nd;
	    }
	  else if(nd>=5 && nd<10)
	    {
	      nd_group=6;
	    }
	  else
	    {
	      nd_group=10;
	    }
	  
	  for(int id=0; id<ND; id++)
	    {
	      if(policy_solved_flag[id]==0 && v_sim[id][jf][kt]>1.0e-10)
		{
		  fprintf(file,"FIRM%d,%.3s,%d,%0.16f,%0.16f,%0.16f,%0.16f,%d,%d,%d,%d\n",
			  jf,name[id],kt,L[id],Y[id],1.0/tau_hat[id],
			  v_sim[id][jf][kt],
			  ix_sim[jf],iz_sim[id][jf][kt],nd,nd_group);
		}
	    }
	}
    }

  fclose(file);

  time(&stop);

  if(verbose)
    printf("Panel data construction complete in %0.0f seconds.\n",difftime(stop,start));
}


///////////////////////////////////////////////////////////////////////////////
// 6. Transition dynamics
///////////////////////////////////////////////////////////////////////////////

#ifdef _MODEL_MAIN

const double dist_tol = 1.0e-11;
const int max_dist_iter = 5000;

double dist[ND][NX][NZ][3] = {{{{0.0}}}};
double tmp_dist[ND][NX][NZ][3] = {{{{0.0}}}};
double tmp_dist2[ND][NX][NZ][3] = {{{{0.0}}}};
int tmp_gex[NT][ND][NX][NZ][3] = {{{{{0}}}}};
double tmp_V[ND][NX][NZ][3] = {{{{0.0}}}};

int temp_shock_periods = 5;
double tr_tau[ND][NT+1];
double tr_exports[ND][NT+1];
double tr_expart[ND][NT+1];
double tr_te[ND][NT+1];

// initialize distribution
void init_dist(int id)
{
  double sum=0.0;
  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  dist[id][ix][iz][0] = x_probs[ix] * z_ucond_probs[iz];
	  sum += dist[id][ix][iz][0];
	}
    }
  if(fabs(sum-1.0)>1.0e-8)
    {
      printf("\nInitial distribution does not sum to one! id = %d, sum = %0.4g\n",id,sum);
    }
}

// distribution iteration driver
int update_dist(int id, double new_dist[NX][NZ][3], double * maxdiff, int *ixs, int *izs, int *iss)
{
  double exit_measure=0.0;

  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  for(int is=0; is<3; is++)
	    {
	      new_dist[ix][iz][is]=0.0;
	    }
	}
    }

  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  double surv_prob = delta[ix][iz];
	  
	  for(int is=0; is<3; is++)
	    {
	      exit_measure += dist[id][ix][iz][is]*(1.0-surv_prob);
	      int igexp = gex[id][ix][iz][is];
	      
	      for(int izp=0; izp<NZ; izp++)
		{
		  new_dist[ix][izp][0] += (1.0-surv_prob)*
		    dist[id][ix][iz][is]*z_ucond_probs[izp];

		  if(igexp==1)
		    {
		      if(is==0)
			{
			  new_dist[ix][izp][1] += dist[id][ix][iz][is]*
			    surv_prob*z_trans_probs[iz][izp];
			}
		      else if(is==1)
			{
			  new_dist[ix][izp][1] += dist[id][ix][iz][is]*
			    surv_prob*z_trans_probs[iz][izp]*rho0;
			  
			  new_dist[ix][izp][2] += dist[id][ix][iz][is]*
			    surv_prob*z_trans_probs[iz][izp]*(1.0-rho0);
			}
		      else if(is==2)
			{
			  new_dist[ix][izp][1] += dist[id][ix][iz][is]*
			    surv_prob*z_trans_probs[iz][izp]*(1.0-rho1);
		      
			  new_dist[ix][izp][2] += dist[id][ix][iz][is]*
			    surv_prob*z_trans_probs[iz][izp]*rho1;
			}
		    }
		  else
		    {
		      new_dist[ix][izp][0] += surv_prob*
			dist[id][ix][iz][is]*z_trans_probs[iz][izp];

		    }

		}
	    }
	}
    }

  double sum = 0.0;
  *maxdiff = 0.0;
  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  for(int is=0; is<3; is++)
	    {
	      sum = sum+new_dist[ix][iz][is];
	      if(fabs(new_dist[ix][iz][is]-dist[id][ix][iz][is])>*maxdiff)
		{
		  *maxdiff = fabs(new_dist[ix][iz][is]-dist[id][ix][iz][is]);
		  *ixs=ix;
		  *izs=iz;
		  *iss=is;
		}
	    }
	}
    }

  if(fabs(sum-1.0)>1.0e-8)
    {
      printf("\nUpdated distribution does not sum to one! id = %d, sum = %0.16f\n",id,sum);
      return 1;
    }
  
  return 0;
}

// distribution iteration loop
int stat_dist(int id)
{
  time_t start, stop;
  int iter=0;
  double maxdiff=999;
  int ixs, izs, iss;
  int status=0;

  time(&start);

  init_dist(id);

  do
    {
      iter++;
      status = update_dist(id,tmp_dist[id],&maxdiff,&ixs,&izs,&iss);
      memcpy(dist[id],tmp_dist[id],NX*NZ*3*sizeof(double));
      
      if(status)
	{
	  printf("Error iterating distribution! id = %d\n",id);
	  break;
	}
    }
  while(maxdiff>dist_tol && iter < max_dist_iter);

  time(&stop);

  if(iter==max_dist_iter)
    {
      status=1;
      printf("Distribution iteration failed! id = %d, ||H1-H0|| = %0.4g, loc = (%d, %d, %d)\n",id,maxdiff,ixs,izs,iss);
    }
  else if(verbose==2)
    printf("Distribution converged for id = %d, iter = %d, ||H1-H0|| = %0.4g\n",id,iter,maxdiff);

  return status;
}

int stat_dist_all_dests()
{
  if(verbose)
    printf("Solving stationary distribtions for all destinations...\n");

  int error=0;
  
  time_t start, stop;
  time(&start);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int id=0; id<ND; id++)
    {      
      if(policy_solved_flag[id]==0)
	{
	  if(stat_dist(id))
	    error=1;
	}
    }

  time(&stop);

  if(verbose)
    printf("Finished stationary distributions in %0.0f seconds.\n",difftime(stop,start));

  return error;
}


void calc_tr_dyn(int id, double tr_moments[2])
{
  double expart_rate=0.0;
  double total_exports=0.0;
  
  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  for(int is=0; is<3; is++)
	    {
	      if(gex[id][ix][iz][is] && dist[id][ix][iz][is]>1.0e-10)
		{
		  expart_rate += dist[id][ix][iz][is];
		  double v = theta*theta_hat*L[id]*Y[id]*tau_hat[id]*x_hat[ix]*z_hat[iz];
		  if(is<2)
		    v = v*pow(xi,1.0-theta);
		  
		  total_exports += dist[id][ix][iz][is] * v;
		}
	    }
	}
    }

  tr_moments[0] = total_exports;
  tr_moments[1] = expart_rate;
  
  return;
}

int tr_dyn_perm_tau_chg(int id, double chg)
{
  double tau_hat0 = tau_hat[id];

  memcpy(tmp_dist[id],dist[id],sizeof(double)*NX*NZ*3);
  memcpy(tmp_gex[0][id],gex[id],sizeof(int)*NX*NZ*3);
  memcpy(tmp_V[id],V[id],sizeof(int)*NX*NZ*3);
  
  double tr_moments[2];

  // period 0: initial steady state
  calc_tr_dyn(id,tr_moments);
  tr_tau[id][0] = pow(tau_hat[id],1.0/(1.0-theta));
  tr_exports[id][0] = tr_moments[0];
  tr_expart[id][0] = tr_moments[1];
  tr_te[id][0] = 0.0;

  // period 1: trade cost changes after firms have made their mkt pen decisions
  tau_hat[id]  = tau_hat0*pow(1.0+chg,1.0-theta);
  calc_tr_dyn(id,tr_moments);  
  tr_tau[id][1] = pow(tau_hat[id],1.0/(1.0-theta));
  tr_exports[id][1] = tr_moments[0];
  tr_expart[id][1] = tr_moments[1];
  tr_te[id][1] = -log(tr_exports[id][1]/tr_exports[id][0])/log(1.0+chg);

  // period 2 onward: new decision rules
  pi_hat[id] = theta_hat * L[id] * Y[id] * tau_hat[id];
  if(solve_policies(id))
    {
      printf("Error solving policy function!\n");
      return 1;
    }
  
  int t;
  for(t=1; t<NT; t++)
    {      
      calc_tr_dyn(id,tr_moments);
      tr_tau[id][t+1] = pow(tau_hat[id],1.0/(1.0-theta));
      tr_exports[id][t+1] = tr_moments[0];
      tr_expart[id][t+1] = tr_moments[1];
      tr_te[id][t+1] = -log(tr_exports[id][t+1]/tr_exports[id][0])/log(1.0+chg);
      
      double junk = 0.0;
      int junk2,junk3,junk4;
      if(update_dist(id, tmp_dist2[id], &junk, &junk2, &junk3, &junk4))
	{
	  printf("Error updating distribution!\n");
	  return 1;
	}
      memcpy(dist[id],tmp_dist2[id],NX*NZ*3*sizeof(double));
    }

  // go back to benchmark trade costs, policies, and dist
  memcpy(dist[id],tmp_dist[id],sizeof(double)*NX*NZ*3);
  memcpy(gex[id],tmp_gex[0][id],sizeof(int)*NX*NZ*3);
  memcpy(V[id],tmp_V[id],sizeof(int)*NX*NZ*3);
  tau_hat[id] = tau_hat0;
  pi_hat[id] = theta_hat * L[id] * Y[id] * tau_hat[id];

  if(verbose==2)
    printf("\tTransition dynamics complete for id=%id!\n",id);
  
  return 0;
}

int tr_dyn_perm_tau_chg_all_dests(double chg)
{
  printf("Analyzing effects of permanent trade cost change of %0.3f...\n",chg);

  time_t start, stop;
  time(&start);
	       
  //int error=0;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int id=0; id<ND; id++)
    {
      if(policy_solved_flag[id]==0)
	{
	  if(tr_dyn_perm_tau_chg(id,chg))
	    policy_solved_flag[id]=1;
	}
    }
  
  time(&stop);
  printf("Complete! Time: %0.0f seconds.\n",difftime(stop,start));

  return 0;
}

int tr_dyn_perm_tau_chg_uncertain(int id, double chg)
{
  double tau_hat0 = tau_hat[id];

  memcpy(tmp_dist[id],dist[id],sizeof(double)*NX*NZ*3);
  memcpy(tmp_gex[0][id],gex[id],sizeof(int)*NX*NZ*3);
  memcpy(tmp_V[id],V[id],sizeof(double)*NX*NZ*3);
  
  double tr_moments[2];

  // period 0: initial steady state
  calc_tr_dyn(id,tr_moments);
  tr_tau[id][0] = pow(tau_hat[id],1.0/(1.0-theta));
  tr_exports[id][0] = tr_moments[0];
  tr_expart[id][0] = tr_moments[1];
  tr_te[id][0] = 0.0;

  // period 1: trade cost changes after firms have made their mkt pen decisions
  tau_hat[id]  = tau_hat0*pow(1.0+chg,1.0-theta);
  calc_tr_dyn(id,tr_moments);  
  tr_tau[id][1] = pow(tau_hat[id],1.0/(1.0-theta));
  tr_exports[id][1] = tr_moments[0];
  tr_expart[id][1] = tr_moments[1];
  tr_te[id][1] = -log(tr_exports[id][1]/tr_exports[id][0])/log(1.0+chg);

  // period 2 onward: new decision rules
  // first solve new steady state policies with 100% probability of new trade costs 
  pi_hat[id] = theta_hat * L[id] * Y[id] * tau_hat[id];
  if(solve_policies(id))
    {
      printf("Error solving policy function!\n");
      return 1;
    }

  // now copy expected continuation value (50% chance of keeping new trade costs, 50% chance of going back)
  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  V[id][ix][iz][0] = 0.5*V[id][ix][iz][0] + 0.5*tmp_V[id][ix][iz][0];
	  V[id][ix][iz][1] = 0.5*V[id][ix][iz][1] + 0.5*tmp_V[id][ix][iz][1];
	  V[id][ix][iz][2] = 0.5*V[id][ix][iz][2] + 0.5*tmp_V[id][ix][iz][2];
	}
    }

  // now iterate once more on policy function to find new policies
  double junk=0.0;
  int junk2[3];
  iterate_policies(id,&junk,junk2);
  
  int t;
  for(t=1; t<NT; t++)
    {      
      calc_tr_dyn(id,tr_moments);
      tr_tau[id][t+1] = pow(tau_hat[id],1.0/(1.0-theta));
      tr_exports[id][t+1] = tr_moments[0];
      tr_expart[id][t+1] = tr_moments[1];
      tr_te[id][t+1] = -log(tr_exports[id][t+1]/tr_exports[id][0])/log(1.0+chg);
      
      double junk = 0.0;
      int junk2,junk3,junk4;
      if(update_dist(id, tmp_dist2[id], &junk, &junk2, &junk3, &junk4))
	{
	  printf("Error updating distribution!\n");
	  return 1;
	}
      memcpy(dist[id],tmp_dist2[id],NX*NZ*3*sizeof(double));
    }

  // go back to benchmark trade costs, policies, and dist
  memcpy(dist[id],tmp_dist[id],sizeof(double)*NX*NZ*3);
  memcpy(gex[id],tmp_gex[0][id],sizeof(int)*NX*NZ*3);
  memcpy(V[id],tmp_V[id],sizeof(double)*NX*NZ*3);
  tau_hat[id] = tau_hat0;
  pi_hat[id] = theta_hat * L[id] * Y[id] * tau_hat0;

  if(verbose==2)
    printf("\tTransition dynamics complete for id=%id!\n",id);
  
  return 0;
}

int tr_dyn_perm_tau_chg_uncertain_all_dests(double chg)
{
  printf("Analyzing effects of permanent trade cost change of %0.3f with 50pct chance of reversion...\n",chg);

  time_t start, stop;
  time(&start);
	       
  //int error=0;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int id=0; id<ND; id++)
    {
      if(policy_solved_flag[id]==0)
	{
	  if(tr_dyn_perm_tau_chg_uncertain(id,chg))
	    policy_solved_flag[id]=1;
	}
    }
  
  time(&stop);
  printf("Complete! Time: %0.0f seconds.\n",difftime(stop,start));

  return 0;
}

int tr_dyn_rer_shock(int id, double shock, double rho)
{
  double tau_hat0 = tau_hat[id];

  memcpy(tmp_dist[id],dist[id],sizeof(double)*NX*NZ*3);
  memcpy(tmp_gex[0][id],gex[id],sizeof(int)*NX*NZ*3);
  memcpy(tmp_V[id],V[id],sizeof(int)*NX*NZ*3);
  
  double tr_moments[2];

  // first solve policies backwards
  for(int t=NT-1; t>=1; t--)
    {
      double rer = exp(shock*pow(rho,t));
      tau_hat[id] = tau_hat0 * pow(rer,theta-1.0);
      pi_hat[id] = theta_hat * L[id] * Y[id] * tau_hat[id];
      if(fabs(tau_hat[id]-tau_hat0)>1.0e-5)
	{	  
	  double junk=0;
	  int junk2[3];
	  iterate_policies(id,&junk,junk2);
	  memcpy(tmp_gex[t][id],gex[id],sizeof(int)*NX*NZ*3);
	}
      else
	{
	  memcpy(tmp_gex[t][id],tmp_gex[0][id],sizeof(int)*NX*NZ*3);
	}
    }

  // now iterate forward

  // period 0: initial steady state
  memcpy(gex[id],tmp_gex[0][id],sizeof(int)*NX*NZ*3);
  tau_hat[id] = tau_hat0;
  calc_tr_dyn(id,tr_moments);
  tr_tau[id][0] = pow(tau_hat[id],1.0/(1.0-theta));
  tr_exports[id][0] = tr_moments[0];
  tr_expart[id][0] = tr_moments[1];
  tr_te[id][0] = 0.0;

  // period 1: trade cost changes after firms have made their mkt pen decisions
  tau_hat[id] = tau_hat0 * pow(exp(shock),theta-1.0);
  calc_tr_dyn(id,tr_moments);  
  tr_tau[id][1] = pow(tau_hat[id],1.0/(1.0-theta));
  tr_exports[id][1] = tr_moments[0];
  tr_expart[id][1] = tr_moments[1];
  tr_te[id][1] = -log(tr_exports[id][1]/tr_exports[id][0])/(-log(1.0+shock));

  // period 2 onward: new decision rules  
  int t;
  for(t=1; t<NT; t++)
    {
      memcpy(gex[id],tmp_gex[t][id],NX*NZ*3*sizeof(int));
      double rer = exp(shock*pow(rho,t));
      tau_hat[id] = tau_hat0 * pow(rer,theta-1.0);
      pi_hat[id] = theta_hat * L[id] * Y[id] * tau_hat[id];
      calc_tr_dyn(id,tr_moments);
      tr_tau[id][t+1] = pow(tau_hat[id],1.0/(1.0-theta));
      tr_exports[id][t+1] = tr_moments[0];
      tr_expart[id][t+1] = tr_moments[1];
      tr_te[id][t+1] = -log(tr_exports[id][t+1]/tr_exports[id][0])/(-log(rer));
      
      double junk = 0.0;
      int junk2,junk3,junk4;
      if(update_dist(id, tmp_dist2[id], &junk, &junk2, &junk3, &junk4))
	{
	  printf("Error updating distribution!\n");
	  return 1;
	}
      memcpy(dist[id],tmp_dist2[id],NX*NZ*3*sizeof(double));
    }

  // go back to benchmark trade costs, policies, and dist
  memcpy(dist[id],tmp_dist[id],sizeof(double)*NX*NZ*3);
  memcpy(gex[id],tmp_gex[0][id],sizeof(int)*NX*NZ*3);
  memcpy(V[id],tmp_V[id],sizeof(int)*NX*NZ*3);
  tau_hat[id] = tau_hat0;
  pi_hat[id] = theta_hat * L[id] * Y[id] * tau_hat[id];

  if(verbose==2)
    printf("\tTransition dynamics complete for id=%id!\n",id);
  
  return 0;
}

int tr_dyn_rer_shock_all_dests(double shock, double rho)
{
  printf("Analyzing effects of temporary RER shock of (%0.3f,%0.3f)...\n",
	 shock,rho);

  time_t start, stop;
  time(&start);
	       
  //int error=0;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int id=0; id<ND; id++)
    {
      if(policy_solved_flag[id]==0)
	{
	  if(tr_dyn_rer_shock(id,shock,rho))
	    policy_solved_flag[id]=1;
	}
    }
  
  time(&stop);
  printf("Complete! Time: %0.0f seconds.\n",difftime(stop,start));

  return 0;

}

int write_tr_dyn_results(const char * fname)
{
  FILE * file = fopen(fname,"w");
  if(file)
    {
      fprintf(file,"d,popt,gdppc,t,tau,exports,expart_rate,trade_elasticity\n");
      for(int id=0; id<ND; id++)
	{
	  if(policy_solved_flag[id]==0)
	    {
	      for(int it=0; it<NT+1; it++)
		{
		  fprintf(file,"%.3s,%0.16f,%0.16f,%d,%0.16f,%0.16f,%0.16f,%0.16f\n",
			  name[id],L[id],Y[id],it,tr_tau[id][it],
			  tr_exports[id][it],tr_expart[id][it],tr_te[id][it]);
		}
	    }
	}
      fclose(file);
      return 0;
    }
  else
    {
      return 1;
    }
}

///////////////////////////////////////////////////////////////////////////////
// 8. Main function and wrappers for non-calibration exercises
///////////////////////////////////////////////////////////////////////////////

int setup()
{
  printf("\nExport Market Penetration Dynamics, Joseph Steinberg, University of Toronto\n");
  printf("Risky new exporter dynamics model\n\n");
  
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
  
  return 0;
}

int benchmark()
{
  printf("Solving and simulating model under benchmark parameterization...\n");

  time_t start, stop;
  time(&start);

  if(solve_policies_all_dests())
    return 1;

  double expart_rate=0.0;
  double exit_rate=0.0;
  double avg_nd = 0.0;
  double new_size;
  simul_all_dests(&expart_rate,&exit_rate,&avg_nd,&new_size);

  if(alpha_flag==0)
    create_panel_dataset("output/acr_microdata.csv");
  else if(alpha_flag==1)
    create_panel_dataset("output/acr2_microdata.csv");
  
  time(&stop);
  printf("\nBenchmark analysis complete! Runtime = %0.0f seconds.\n",
	 difftime(stop,start));
	  
  return 0;
}

int main(int argc, char * argv[])
{
  time_t start, stop;
  time(&start);

  alpha_flag=0;
  if(argc==1 || (argc>1 && strcmp(argv[1],"0")==0))
    {
      alpha_flag=0;
    }
  else if(argc>1 && strcmp(argv[1],"1")==0)
    {
      alpha_flag=1;
    }

  // setup environment
  linebreak();    
  if(setup())
      return 1;

  // solve and simulate model under benchmark calibration
  linebreak();	  
  if(benchmark())
      return 1;

  // solve stationary distributions
  linebreak();
  if(stat_dist_all_dests())
    return 1;

  
  // effects of permanent drop in trade costs
  linebreak();  
  if(tr_dyn_perm_tau_chg_all_dests(-0.1))
    return 1;

  if(alpha_flag==0)
    {
      if(write_tr_dyn_results("output/tr_dyn_perm_tau_drop_acr.csv"))
	return 1;
    }
  else if(alpha_flag==1)
    {
      if(write_tr_dyn_results("output/tr_dyn_perm_tau_drop_acr2.csv"))
	return 1;
    }
  

  // effects of permanent drop in trade costs with uncertainty
  linebreak();  
  if(tr_dyn_perm_tau_chg_uncertain_all_dests(-0.1))
    return 1;

  if(alpha_flag==0)
    {
      if(write_tr_dyn_results("output/tr_dyn_perm_tau_drop_uncertain_acr.csv"))
	return 1;
    }
  else if(alpha_flag==1)
    {
      if(write_tr_dyn_results("output/tr_dyn_perm_tau_drop_uncertain_acr2.csv"))
	return 1;
    }


  // effects of temporary good depreciation
  double shock = log(1.0+(theta-1.0)/10.0)/(theta-1.0);
  linebreak();
  if(tr_dyn_rer_shock_all_dests(shock,0.75))
    return 1;

  if(alpha_flag==0)
    {
      if(write_tr_dyn_results("output/tr_dyn_rer_dep_acr.csv"))
	return 1;
    }
  else if(alpha_flag==1)
    {
      if(write_tr_dyn_results("output/tr_dyn_rer_dep_acr2.csv"))
	return 1;
    }

    //free_cost_spline_mem();


  // finish program
  linebreak();  
  time(&stop);
  printf("\nProgram complete! Total runtime: %0.16f seconds.\n",difftime(stop,start));

  return 0;
}
#endif

