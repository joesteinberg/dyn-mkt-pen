///////////////////////////////////////////////////////////////////////////////
// DYN_MKT_PEN.C
// Joseph B. Steinberg, University of Toronto
//
// This program performs all of the quantitative analyses in the paper 
// "Export Market Penetration Dynamics." The code is organized into sections:
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
//	6. Main function
//
// To compile the program, the user must have the following libraries:
// 	i. GNU GSL library (I have used version 2.1)
//	ii. OpenMP
// There is a makefile in the same directory as the source code. To compile
// and run the program, simply type "make" followed by "./dyn_mkt_pen".


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
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_linalg.h>
#include <omp.h>

// macros: discretization
#define NX 50 // fixed-effect grid size
#define NZ 101 // productivity shock grid size
#define ND 63 // number of destinations
#define NM 100 // dynamic policy function grid size
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
const double policy_tol_rel = 1.0e-3;
const double policy_tol_abs = 1.0e-4;
const int policy_max_iter = 150;
const double endo_grid_tol = 1.0e-9;
const int endo_grid_max_iter = 1000;
const double m_grid_ub = 0.999;
const double m_grid_exp = 1.1;
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
double alpha_n = 0.0; // returns to scale in marketing to new customers
double gamma_n = 0.0; // diminishing returns to scale in marketing to new customers
double psi_n = 0.0; // marketing efficiency for new customers
double alpha_o = 0.0; // returns to scale in marketing to old customers
double gamma_o = 0.0; // diminishing returns to scale in marketing to old customers
double psi_o = 0.0; // marketing efficiency for old customers
double z_grid_mult_lb = 0.0;
double z_grid_mult_ub = 0.0;

// customer base grid
double m_grid[NM] = {0.0};

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
//double tau[ND] = {0.0}; // trade cost
double La_n[ND] = {0.0}; // = L^(alpha_n)
double Lam_n[ND] = {0.0}; // = L^(alpha_n-1)
double La_o[ND] = {0.0}; // = L^(alpha_o)
double Lam_o[ND] = {0.0}; // = L^(alpha_o-1)
double tau_hat[ND] = {0.0}; // = tau^(1-theta)
double pi_hat[ND] = {0.0}; // theta_hat*L*Y*tau_hat

// law of motion for market penetration
static inline double mp(double m, double n, double o)
{
  return (1-m)*n + m*o;
}

// exports
static inline double exports(int id, int ix, int iz, int im)
{
  return theta*theta_hat*m_grid[im]*L[id]*Y[id]*tau_hat[id]*x_hat[ix]*z_hat[iz];
}

// profits
static inline double profits(int id, int ix, int iz, int im)
{
  return theta_hat*m_grid[im]*L[id]*Y[id]*tau_hat[id]*x_hat[ix]*z_hat[iz];
}

// marketing cost for new customers
static inline double s(int id, double m, double n)
{
  if(n<(1.0-m))
    {
      return pow((1.0-m)*L[id],alpha_n) * ( 1 - pow((1.0-m-n)/(1.0-m),1.0-gamma_n) ) / psi_n / (1.0-gamma_n);
    }
  else
    {
      return GSL_NAN;
    }
}

// marketing cost for old customers
static inline double r(int id, double m, double o)
{
  if(o<m)
    {
      return pow(m*L[id],alpha_o) * ( 1 - pow((m-o)/m,1.0-gamma_o) ) / psi_o / (1.0-gamma_o);
    }
  else
    {
      return GSL_NAN;
    }
}

// derivative of s wrt n
static inline double ds_dn(int id, double m, double n)
{
  if(n<(1.0-m))
    {
      return pow((1.0-m)*L[id],alpha_n) / psi_n / (1.0-m) / pow((1.0-m-n)/(1.0-m),gamma_n);
    }
  else
    {
      return GSL_NAN;
    }
}

static inline double ds_dm(int id, double m, double n)
{
  if(n<(1.0-m))
    {
      double tmp1 = pow((1.0-m)*L[id], alpha_n)/psi_n/(1.0-gamma_n);
      double d_tmp1 = -alpha_n * pow(1.0-m,alpha_n-1.0) * pow(L[id],alpha_n) / psi_n / (1.0-gamma_n);
      double tmp2 = -pow( (1.0-m-n)/(1.0-m) , 1.0-gamma_n );
      double d_tmp2 = (1.0-gamma_n)*(pow(1.0-m-n,-gamma_n)*pow(1.0-m,gamma_n-1.0) - pow(1.0-m-n,1.0-gamma_n)*pow(1.0-m,gamma_n-2.0));

      double retval = d_tmp1 + tmp1*d_tmp2 + tmp2*d_tmp1;
  
      return retval;
	
    }
  else
    {
      return GSL_NAN;
    }
}

// derivative of r wrt o
static inline double dr_do(int id, double m, double o)
{
  if(o<m)
    {
      return pow(m*L[id],alpha_o) / psi_o / m / pow((m-o)/m,gamma_o);
    }
  else
    {
      return GSL_NAN;
    }
}

static inline double dr_dm(int id, double m, double o)
{
  if(o<m)
    {
      double tmp1 = pow(m*L[id], alpha_o)/psi_o/(1.0-gamma_o);
      double d_tmp1 = alpha_o * pow(m,alpha_o-1.0) * pow(L[id],alpha_o) / psi_o / (1.0-gamma_o);
      double tmp2 = -pow( (m-o)/m , 1.0-gamma_o );
      double d_tmp2 = -(1.0-gamma_o)*(pow(m-o,-gamma_o)*pow(m,gamma_o-1.0) - pow(m-o,1.0-gamma_o)*pow(m,gamma_o-2.0));

      double retval = d_tmp1 + tmp1*d_tmp2 + tmp2*d_tmp1;
  
      return retval;
    }
  else
    {
      return GSL_NAN;
    }
}

// discretize market penetration grid
void discretize_m()
{
  expspace(0.0,m_grid_ub,NM,m_grid_exp,m_grid);
  return;
}

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
      double m[NX-1];
      for(int i=0; i<NX-1; i++)
	{
	  if(i<NX-1)
	    m[i] = gsl_cdf_ugaussian_Pinv( ((double)(i+1))/((double)(NX)) ) * kappa_x;
	  
	  x_probs[i] = 1.0/NX;

	  if(i==0)
	    x_cumprobs[i] = x_probs[i];
	  else
	    x_cumprobs[i] = x_cumprobs[i-1] + x_probs[i];
	}

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
  Q = 0.82457545;
  delta0 = 10.0;
  delta1 = 0.02;
  theta = 5.0;
  theta_hat = (1.0/theta) * pow(theta/(theta-1.0),1.0-theta);
  kappa_x = 0.68639679;
  sig_z = 0.44408646;
  rho_z = 0.71221072;
  alpha_n = 0.2;
  alpha_o = 0.8;
  gamma_n = 6.0;
  gamma_o = 2.0;
  psi_n = 0.3;
  psi_o = 0.2;
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
      char buffer[128];
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
	      Y[id] = gdppc;
	      tau_hat[id] = 1.0/tau_;
	      strncpy(name[id],buffer,3);
	      //tau_hat[id] = pow(tau[id],1.0-theta);
	      La_n[id] = pow(L[id],alpha_n);
	      Lam_n[id] = pow(L[id],alpha_n-1.0);
	      La_o[id] = pow(L[id],alpha_o);
	      Lam_o[id] = pow(L[id],alpha_o-1.0);
	      pi_hat[id] = theta_hat * L[id] * Y[id] * tau_hat[id];
	    }
	}

      return 0;
    }
}


///////////////////////////////////////////////////////////////////////////////
// 3. Exporting cost c(m,m')
///////////////////////////////////////////////////////////////////////////////

double m_grid[NM];
double export_cost[ND][NM][NM] = {{{0.0}}}; // f(m,m') = min{ s(n,m)+r(o,m) s.t. n + o = m' && n in [0,1-m] && o in [0,m] }
double export_cost_argmin_n[ND][NM][NM] = {{{0.0}}}; // argmin[0]
double export_cost_argmin_o[ND][NM][NM] = {{{0.0}}}; // argmin[1]
double export_cost_deriv_m[ND][NM][NM] = {{{0.0}}}; // f_1(m,m') NOTE: grid ordering is [id][im][imp] for interpolation purposes
double export_cost_deriv_mp[ND][NM][NM] = {{{0.0}}}; // f2_m,m') NOTE: grid ordering is [id][im][imp] for interpolation purposes
double export_cost_deriv_mp_2[ND][NM][NM] = {{{0.0}}}; // f2_m,m') NOTE: grid ordering is [id][imp][im] for interpolation purposes

double export_cost_min_foc(double n, void * p)
{
  int id = ((int *)p)[0];
  int im = ((int *)p)[1];
  int imp = ((int *)p)[2];

  double m = m_grid[im];
  double mp = m_grid[imp];
  double o = mp-n;

  double retval = ds_dn(id,m,n) - dr_do(id,m,o);
  if(n>1.0-m || gsl_isnan(retval) || gsl_isinf(retval))
    {
      return GSL_NAN;
      printf("NAN!");
      exit(1);
    }
  return retval;
}

int solve_export_cost(int id)
{
  //if(verbose)
  //  printf("\tSolving static export cost minimization problem for id=%d...\n",id);

  time_t stop, start;
  time(&start);

  for(int im=0; im<NM; im++)
    {
      double m = m_grid[im];
      
      for(int imp=0; imp<NM; imp++)
	{
	  double mp = m_grid[imp];
	  
	  if(im==0) // new entrants can only attract new customers
	    {
	      export_cost_argmin_n[id][im][imp]=mp;
	      export_cost_argmin_o[id][im][imp]=0.0;
	      export_cost[id][im][imp] = s(id,m,mp);
	    }
	  else if(imp==0)
	    {
	      export_cost_argmin_n[id][im][imp]=0.0;
	      export_cost_argmin_o[id][im][imp]=0.0;
	      export_cost[id][im][imp]=0.0;
	    }
	  else
	    {
	      int p[3] = {id,im,imp};
	      double lb = fmax(0.0,mp-m);
	      double ub = fmin(1.0,fmin(1.0-m,mp));

	      // if marginal cost of aquiring new customers is higher than marginal cost of keeping old ones
	      // even when n=0 and o=m'/m, then corner solution n=lb is best
	      //if(mp<m-1.0e-8 && export_cost_min_foc(0.0,&p) < export_cost_min_foc(1.0e-8,&p))
	      if(export_cost_min_foc(lb+1.0e-11,&p) > 0.0)
		{
		  export_cost_argmin_n[id][im][imp]=lb;
		  export_cost_argmin_o[id][im][imp]=mp-lb;
		  export_cost[id][im][imp] = s(id,m,lb+1.0e-11) + r(id,m,mp-lb-1.0e-11);
		}
	      // if marginal cost of aquiring new customers is lower than marginal cost of keeping old ones
	      // even when n=m' and o=0, then corner solution with o=0 is best
	      else if(export_cost_min_foc(ub-1.0e-11,&p) < 0.0)
		{
		  export_cost_argmin_n[id][im][imp]=ub;
		  export_cost_argmin_o[id][im][imp]=mp-ub;
		  export_cost[id][im][imp] = s(id,m,ub-1.0e-11) + r(id,m,mp-ub+1.0e-11);
		}
	      else // interior solution
		{
		  gsl_function f;
		  f.function = &export_cost_min_foc;
		  f.params=&p;

		  export_cost_argmin_n[id][im][imp] = (ub+lb)/2.0;
		  if(find_root_1d(&f,lb+1.0e-11,ub-1.0e-11,&(export_cost_argmin_n[id][im][imp])))
		    {
		      printf("\tError solving static first-order condition! (id,im,imp) = (%d,%d,%d)\n",id,im,imp);
		      return 1;
		    }
		  export_cost_argmin_o[id][im][imp] = mp-export_cost_argmin_n[id][im][imp];
		  export_cost[id][im][imp] = s(id,m,export_cost_argmin_n[id][im][imp]) + r(id,m,export_cost_argmin_o[id][im][imp]);
		}
	    }
	}
    }

  for(int im=0; im<NM; im++)
    {
      for(int imp=0; imp<NM; imp++)
	{
	  double dmp=0.0;
	  if(imp==0)
	    {
	      dmp = (export_cost[id][im][imp+1]-export_cost[id][im][imp])/
		(m_grid[imp+1]-m_grid[imp]);
	    }
	  else if(imp==NM-1)
	    {
	      dmp = (export_cost[id][im][imp]-export_cost[id][im][imp-1])/
		(m_grid[imp]-m_grid[imp-1]);
	    }
	  else
	    {
	      dmp =0.5*(export_cost[id][im][imp]-export_cost[id][im][imp-1])/
		(m_grid[imp]-m_grid[imp-1]) +
		0.5*(export_cost[id][im][imp+1]-export_cost[id][im][imp])/
		(m_grid[imp+1]-m_grid[imp]);
	    }
	  export_cost_deriv_mp[id][im][imp] = dmp;
	  export_cost_deriv_mp_2[id][imp][im] = dmp;

	  double dm=0.0;
	  if(im==0)
	    {
	      dm = (export_cost[id][im+1][imp]-export_cost[id][im][imp])/
		(m_grid[im+1]-m_grid[im]);
	    }
	  else if(im==NM-1)
	    {
	      dm = (export_cost[id][im][imp]-export_cost[id][im-1][imp])/
		(m_grid[im]-m_grid[im-1]);
	    }
	  else
	    {
	      dm = 0.5*(export_cost[id][im][imp]-export_cost[id][im-1][imp])/
		(m_grid[im]-m_grid[im-1]) +
		0.5*(export_cost[id][im+1][imp]-export_cost[id][im][imp])/
		(m_grid[im+1]-m_grid[im]);
	    }
	  export_cost_deriv_m[id][im][imp] = dm;
	}
    }

  time(&stop);

  if(verbose==2)
    printf("\tStatic problem for %.3s completed in %0.0f seconds.\n",name[id],difftime(stop,start));

  return 0;
}

int solve_export_cost_all_dests()
{
  if(verbose)
    printf("\nSolving static export cost minimization problem for all destinations...\n");
  
  time_t start, stop;
  time(&start);

  
  int error=0;
  
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int id=0; id<ND; id++)
    {
      if(!error)
	{
	  int error_th = solve_export_cost(id);
	  if(error_th)
	    {
	      error = 1;
	    }
	  
#ifdef _OPENMP
#pragma omp flush(error)
#endif
	}

    }

  time(&stop);
  
  if(!error && verbose)
    printf("Finished static problems in %0.0f seconds.\n",difftime(stop,start));

  if(error)
    printf("Failed to solve all destinations' static problems! Time = %0.0f seconds.\n",difftime(stop,start));
  
  return error;
  
}

void save_export_cost()
{
  WRITE_VECTOR(m_grid,NM,"output/m_grid.txt");
  WRITE_VECTOR(export_cost,ND*NM*NM,"output/export_cost.txt");
  WRITE_VECTOR(export_cost_argmin_n,ND*NM*NM,"output/export_cost_argmin_n.txt");
  WRITE_VECTOR(export_cost_argmin_o,ND*NM*NM,"output/export_cost_argmin_o.txt");
  WRITE_VECTOR(export_cost_deriv_m,ND*NM*NM,"output/export_cost_deriv_m.txt");
  WRITE_VECTOR(export_cost_deriv_mp,ND*NM*NM,"output/export_cost_deriv_mp.txt");
}

///////////////////////////////////////////////////////////////////////////////
// 3. Dynamic program
///////////////////////////////////////////////////////////////////////////////

// equilibrium objects
double gm[ND][NX][NZ][NM] = {{{{0.0}}}}; // policy function for market penetration
double gm_reform[ND][NX][NZ][NM] = {{{{0.0}}}}; // policy function for market penetration after trade reform
double dV[ND][NX][NZ][NM] = {{{{0.0}}}}; // value function derivative
double expart_rate[ND] = {0.0};
int policy_solved_flag[ND] = {0};

typedef struct
{
  int id;
  int ix;
  int iz;
  int im;
  gsl_interp_accel * acc;
}dp_params;

void init_dp_objs(int id)
{
   for(int ix=NX-1; ix>=0; ix--)
    {
      for(int iz=NZ-1; iz>=0; iz--)
	{
  	  for(int im = NM-1; im>=0; im--)
	    {
	      dV[id][ix][iz][im] = 0.0;
	    }
	}
    }
}



// s_2(0,m') >= Q*sum_{z' in Z}dV_dm(z',m')
double entrant_foc(double mp, void * data)
{
  dp_params * p = (dp_params *)data;
  int id = p->id;
  int ix = p->ix;
  int iz = p->iz;
  gsl_interp_accel * acc = p->acc;

  int imp = gsl_interp_accel_find(acc, m_grid, NM, mp);
  
  double EdV=0.0;
  for(int izp=0; izp<NZ; izp++)
    {
      if(z_trans_probs[iz][izp]>1.0e-11)
	{
	  double Vm = interp_with_ix(m_grid,dV[id][ix][izp],NM,mp,imp);
	  double tmp = z_trans_probs[iz][izp]*Vm;
	  EdV += tmp;

	  // for future states where the firm switches from exiting to staying in
	  // we smooth things out by using interpolation to find
	  /*if(izp>0 &&
	     fabs(EdV)<1.0e-10 &&
	     fabs(Vm) > 1.0e-10)
	    {
	      double Vm2 = interp_with_ix(m_grid,dV[id][ix][izp+1],NM,mp,imp);
	      double vdist = Vm2-Vm;
	      double zdist = z_grid[izp+1]-z_grid[izp];
	      double zcrit = z_grid[izp] - Vm*(zdist/vdist);
	      double frac = gsl_cdf_ugaussian_P( (zcrit + zdist/2.0 - rho_z*z_grid[iz]) / sig_z);
	      double test=0;
	      }*/
	}
    }

  //return ds_dn(id,0.0,mp) - Q*delta*(pi_hat[id]*E_xhat_zhat[ix][iz] + EdV);
  return ds_dn(id,0.0,mp) - pi_hat[id]*x_hat[ix]*z_hat[iz] - Q*delta[ix][iz]*EdV;
}

int iterate_entrant_policy(int id)
{
  gsl_interp_accel * acc = gsl_interp_accel_alloc();
  
  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  gsl_interp_accel_reset(acc);
	  
	  dp_params p = {id,ix,iz,0,acc};
	  double lower_bound=delta_root;
	  double upper_bound = m_grid[NM-1]-delta_root;

	  if(entrant_foc(0.0,&p)>0.0)
	    {
	      gm[id][ix][iz][0] = 0.0;
	    }
	  else if(entrant_foc(m_grid[NM-1],&p)<0.0)
	    {
	      gm[id][ix][iz][0] = m_grid[NM-1];
	    }
	  else
	    {
	      if(iz>0)
		{
		  lower_bound = fmax(lower_bound,gm[id][ix][iz-1][0]-delta_root);
		}
	      else
		{
		  lower_bound = fmax(lower_bound,gm[id][ix-1][iz][0]-delta_root);
		}
	      gsl_function f;
	      f.function = &entrant_foc;
	      f.params=&p;
	      if(find_root_1d(&f,lower_bound,upper_bound,&(gm[id][ix][iz][0])))
		{
		  printf("\nError solving entrant's first-order condition! (id,ix,iz) = (%d,%d,%d)\n",ix,iz,id);
		  gsl_interp_accel_free(acc);
		  return 1;
		}
	    }
	}
    }

  gsl_interp_accel_free(acc);
  return 0;

}

// f_2(m,m') >= Q*sum_{z' in Z}dV_dm(z',m')
double incumbent_foc(double mp, void * data)
{
  dp_params * p = (dp_params *)data;
  int id = p->id;
  int ix = p->ix;
  int iz = p->iz;
  int im = p->im;
  gsl_interp_accel * acc = p->acc;

  int imp = gsl_interp_accel_find(acc, m_grid, NM, mp);
  
  double EdV=0.0;
  for(int izp=0; izp<NZ; izp++)
    {
      if(z_trans_probs[iz][izp]>1.0e-11)
	{
	  EdV += z_trans_probs[iz][izp]*interp_with_ix(m_grid,dV[id][ix][izp],NM,mp,imp);
	}
    }

  double retval = interp_with_ix(m_grid,export_cost_deriv_mp[id][im],NM,mp,imp) -
    //Q*delta*(pi_hat[id]*E_xhat_zhat[ix][iz] + EdV);
    pi_hat[id]*x_hat[ix]*z_hat[iz] - Q*delta[ix][iz]*EdV;
  
  if(gsl_isinf(retval) || gsl_isnan(retval))
    {
      printf("Incumbent FOC is INF or NAN!\n");
    }

  return retval;
}

// endogenous grid method
// first order condition: f_2(m,m') >= Q*sum_{z' in Z}dV_dm(z',m') := EdV(m')
// solve for m using interpolation: treat f_2(:,m') as x-axis, m grid as y-axis
double invert_incumbent_foc(void * data)
{
  dp_params * p = (dp_params *)data;
  int id = p->id;
  int ix = p->ix;
  int iz = p->iz;
  int im = p->im;
  gsl_interp_accel * acc = p->acc;

  double EdV = 0.0;
  for(int izp=0; izp<NZ; izp++)
    {
      EdV += Q*delta[ix][iz]*dV[id][ix][izp][im]*z_trans_probs[iz][izp];
    }

  return interp(acc,export_cost_deriv_mp_2[id][im],m_grid,NM,EdV);
}

// dV_dm(z,m) = L*Y*(x*z)^{theta-1} - f_1(m,m'(z,m))
double envelope_cond(double mp, void * data)
{
  dp_params * p = (dp_params *)data;
  int id = p->id;
  //int ix = p->ix;
  //int iz = p->iz;
  int im = p->im;
  gsl_interp_accel * acc = p->acc;

  //double pi = pi_hat[id]*x_hat[ix]*z_hat[iz];
  double f_1 = interp(acc,m_grid,export_cost_deriv_m[id][im],NM,mp);
  
  //return pi-f_1;
  return -f_1;
}

int iterate_incumbent_policy(int id, double * maxdiff, int imaxdiff[3])
{
  *maxdiff = -HUGE_VAL;
  
  gsl_interp_accel * acc = gsl_interp_accel_alloc();
  
  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  int im=0;
	  for(im=1; im<NM; im++)
	    {
	      dp_params p = {id,ix,iz,im,acc};
	      double lower_bound = 0.0;
	      double upper_bound = m_grid[NM-1]-delta_root;
	      double gm_tmp=0.0;
	      
	      if(incumbent_foc(0.0,&p)>0)
		{
		  gm_tmp = 0.0;
		}
	      else if(incumbent_foc(m_grid[NM-1],&p)<0.0)
		{
		  gm_tmp = m_grid[NM-1];
		}
	      else
		{
		  gsl_function f;
		  f.function = &incumbent_foc;
		  f.params=&p;
		  if(find_root_1d(&f,lower_bound,upper_bound,&gm_tmp))
		    {
		      printf("\nError solving incumbent's first-order condition! (id,ix,iz) = (%d,%d,%d)\n",ix,iz,id);
		      gsl_interp_accel_free(acc);
		      return 1;
		    }
		}
	      
	      double diff = fabs(gm_tmp-gm[id][ix][iz][im]) - gm[id][ix][iz][im]*policy_tol_rel;
	      gm[id][ix][iz][im] = gm_tmp;
	      if(diff>*maxdiff)
		{
		  *maxdiff=diff;
		  imaxdiff[0]=ix;
		  imaxdiff[1]=iz;
		  imaxdiff[2]=im;
		}
	    }
		
	}
	
      for(int iz=0; iz<NZ; iz++)
	{
	  gsl_interp_accel_reset(acc);
	  for(int im=0; im<NM; im++)
	    {
	      dp_params p = {id,ix,iz,im,acc};
	      dV[id][ix][iz][im] = envelope_cond(gm[id][ix][iz][im],&p);
	    }
	}
    }
  
  gsl_interp_accel_free(acc);
  
  return 0;
}

// iteration loop
int solve_policies(int id)
{
  //if(verbose)
  //  printf("\tSolving dynamic program via policy function iteration for id=%d...\n",id);
  
  time_t start, stop;
  time(&start);

  init_dp_objs(id);

  int status = 0;
  double maxdiff = 999;
  int imaxdiff[3] = {0};
  
  int iter=0;
  do
    {
      iter++;
      status = iterate_entrant_policy(id);
      if(status)
	{
	  printf("\tError iterating entrant policy function! id = %d\n",id);
	  break;
	}

      status = iterate_incumbent_policy(id,&maxdiff,imaxdiff);
      if(verbose==3)
	{
	  printf("\t\tIter %d, diff = %0.2g, loc = (%d, %d, %d), gm[loc] = %0.4g\n",
		 iter,maxdiff,imaxdiff[0],imaxdiff[1],imaxdiff[2],
		 gm[id][imaxdiff[0]][imaxdiff[1]][imaxdiff[2]]);
	}
    }
  while(maxdiff>policy_tol_abs && iter < policy_max_iter);

  expart_rate[id] = 0.0;
  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  expart_rate[id] += x_probs[ix]*z_ucond_probs[iz]*(gm[id][ix][iz][0]>1.0e-10);
	}
    }    

  time(&stop);

  if(iter==policy_max_iter)
    {
      status=1;
      if(verbose==2)
	printf("\tPolicy function iteration failed for %.3s! Diff = %0.4g",name[id],maxdiff);
    }
  else
    {
      if(verbose==2)
	{
	  printf("\tPolicy function converged for %.3s in %0.0f seconds!",
		 name[id],difftime(stop,start));
	  printf(" Export participation rate = %0.8f.\n",100*expart_rate[id]);
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

void save_policies()
{
  WRITE_VECTOR(m_grid,NM,"output/m_grid.txt");
  WRITE_VECTOR(gm,ND*NX*NZ*NM,"output/gm.txt");
}

///////////////////////////////////////////////////////////////////////////////
// 4. Simulation
///////////////////////////////////////////////////////////////////////////////

// storage for simulated data
// we use 3*NT to store NT throwaway periods, NT periods to simulate for calibration,
// and NT periods for the shock analysis
unsigned long int seed = 0;
double x_rand[NF];
double z_rand[ND][NF][NT*3];
double surv_rand[NF][NT*3];
int ix_sim[NF];
int iz_sim[ND][NF][NT*3];
double m_sim[ND][NF][NT*3];
double v_sim[ND][NF][NT*3];

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
	  
	  for(int t=0; t<NT*3; t++)
	    {
	      z_rand[id][i][t] = gsl_rng_uniform(r);
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
void simul_benchmark(int id)
{
  //if(verbose)
  //  printf("\n\tSimulating model for id=%d...\n",id);

  time_t start, stop;
  time(&start);

  int max_kt = NT*2;
  //if(post_reform)
  //  max_kt = NT*3;

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
      
      // initialize market penetration to zero
      double m = 0.0;
      m_sim[id][jf][0] = m;
            
      for(int kt=0; kt<max_kt; kt++)
	{
	  if(surv_rand[jf][kt]>delta[ix][iz])
	    {
	      m_sim[id][jf][kt]=0.0;
	      m=0.0;
	      v_sim[id][jf][kt] = -99.9;
	      iz_sim[id][jf][kt+1] = gsl_interp_accel_find(acc1, z_ucond_cumprobs, NZ, z_rand[id][jf][kt+1]);
	    }
	  else
	    {
	      m_sim[id][jf][kt] = interp(acc2,m_grid,gm[id][ix][iz],NM,m);
	      m = m_sim[id][jf][kt];
	      
	      v_sim[id][jf][kt] = theta*theta_hat*L[id]*Y[id]*tau_hat[id]*x_hat[ix]*z_hat[iz]*m_sim[id][jf][kt];
	      if(m_sim[id][jf][kt]<1.0e-8)
		v_sim[id][jf][kt]=-99.9;
	      
	      iz_sim[id][jf][kt+1] = gsl_interp_accel_find(acc1, z_trans_cumprobs[iz], NZ, z_rand[id][jf][kt+1]);
	    }

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

// main simulation function
void simul_permanent_shock(int id)
{
  //if(verbose)
  //  printf("\n\tSimulating model for id=%d...\n",id);

  time_t start, stop;
  time(&start);

  int min_kt=Nt*2;
  int max_kt = NT*3;

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
      
      // initialize market penetration to zero
      double m = 0.0;
      m_sim[id][jf][0] = m;
            
      for(int kt=0; kt<max_kt; kt++)
	{
	  if(surv_rand[jf][kt]>delta[ix][iz])
	    {
	      m_sim[id][jf][kt]=0.0;
	      m=0.0;
	      v_sim[id][jf][kt] = -99.9;
	      iz_sim[id][jf][kt+1] = gsl_interp_accel_find(acc1, z_ucond_cumprobs, NZ, z_rand[id][jf][kt+1]);
	    }
	  else
	    {
	      m_sim[id][jf][kt] = interp(acc2,m_grid,gm[id][ix][iz],NM,m);
	      m = m_sim[id][jf][kt];
	      
	      v_sim[id][jf][kt] = theta*theta_hat*L[id]*Y[id]*tau_hat[id]*x_hat[ix]*z_hat[iz]*m_sim[id][jf][kt];
	      if(m_sim[id][jf][kt]<1.0e-8)
		v_sim[id][jf][kt]=-99.9;
	      
	      iz_sim[id][jf][kt+1] = gsl_interp_accel_find(acc1, z_trans_cumprobs[iz], NZ, z_rand[id][jf][kt+1]);
	    }

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

double simul_all_dests(int scenario)
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
	if(scenario==0)
	  simul_benchmark(id);
    }

  double expart_rate_[NT] = {0.0};
  double avg_expart_rate=0.0;
  for(int kt=NT; kt<NT*2; kt++)
    {
      for(int jf=0; jf<NF; jf++)
	{
	  int exporter=0;
	  
	  for(int id=0; id<ND; id++)
	    {
	      if(exporter==0 && v_sim[id][jf][kt]>1.0e-10)
		{
		  expart_rate_[kt-NT] += 1.0;
		  exporter=1;
		}
	    }
	}
      expart_rate_[kt-NT] = expart_rate_[kt-NT]/NF;
      avg_expart_rate += expart_rate_[kt-NT];
    }

  avg_expart_rate=avg_expart_rate/NT;

  time(&stop);
  
  if(verbose)
    printf("Finished simulations in %0.0f seconds. Overall export participation rate = %0.8f.\n",difftime(stop,start),100*avg_expart_rate);
  
  return avg_expart_rate;
}


void create_panel_dataset_benchmark()
{
  if(verbose)
    printf("\nCreating panel dataset from simulation...\n");

  time_t start, stop;
  time(&start);

  int min_kt = NT;
  int max_kt = NT*2;
  FILE * file;
  
  file = fopen("output/model_microdata_calibration.csv","w");

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
			  jf,name[id],kt,L[id],Y[id],1.0/tau_hat[id],v_sim[id][jf][kt],ix_sim[jf],iz_sim[id][jf][kt],nd,nd_group);
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
// 6. Main function and wrappers for non-calibration exercises
///////////////////////////////////////////////////////////////////////////////

#ifdef _MODEL_MAIN

int setup()
{
  printf("\nExport Market Penetration Dynamics, Joseph Steinberg, University of Toronto\n\n");
  
#ifdef _OPENMP
  omp_set_num_threads(ND/2+1);
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
  discretize_m();
  random_draws();

  return 0;
}

int benchmark()
{
  if(solve_export_cost_all_dests())
      return 1;
  save_export_cost();

  if(solve_policies_all_dests())
    return 1;
  save_policies();
    
  simul_all_dests(0);
  create_panel_dataset_benchmark();

  time(&stop);
  printf("\nCalling python scripts to process and analyze simulated microdata...\n");
  system("python3 -W ignore ../python/model_microdata_prep.py");
  system("python3 -W ignore ../python/sumstats_regs.py");
  printf("\nData processing complete! Time: %0.16f seconds.\n",difftime(stop,start));

  return 0;
}

int permanent_tau_chg(double chg)
{
  for(int id=0; id<NDl id++)
    {
      tau_hat[id] = tau_hat[id]*pow(chg,1.0-theta);
    }
  
  if(solve_export_cost_all_dests())
      return 1;

  if(solve_policies_all_dests())
    return 1;
    
  simul_all_dests(1);
  create_panel_dataset_permanent_shock();

  time(&stop);
  printf("\nCalling python scripts to process and analyze simulated microdata...\n");
  system("python3 -W ignore ../python/model_microdata_prep.py");
  system("python3 -W ignore ../python/sumstats_regs.py");
  printf("\nData processing complete! Time: %0.16f seconds.\n",difftime(stop,start));

  return 0;
}


int main()
{
  //gsl_set_error_handler_off();
  
  time_t start, stop;
  time(&start);

  if(setup())
    {
      return 1;
    }
  
  if(benchmark())
    {
      return 1;
    }
  
  time(&stop);
  printf("\nProgram complete! Total runtime: %0.16f seconds.\n",difftime(stop,start));


  return 0;
}
#endif

