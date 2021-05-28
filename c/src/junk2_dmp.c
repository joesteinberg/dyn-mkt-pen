/////////////////////////////////////////////////////////////////////////////
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
// and run the program, simply type "make" followed by "./dyn_mkt_pen".


/////////////////////////////////////////////////////////////////////////////
// 1. Includes, macros, etc.
/////////////////////////////////////////////////////////////////////////////

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
//#include <gsl/gsl_matrix.h>
//#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
//#include <gsl/gsl_multifit_nlinear.h>
#include <nlopt.h>
#include <omp.h>

// macros: discretization
#define NX 50 // fixed-effect grid size
#define NZ 101// productivity shock grid size
#define ND 63 // number of destinations
#define NM 100 // dynamic policy function grid size
#define NT 100 // simulation length
#define NF 25000 // simulation population size

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
const int policy_max_iter = 100;
const double m_grid_ub = 0.999;
const double m_grid_exp = 1.1;
const double x_grid_ub_mult = 10.0;
const double x_grid_exp = 1.0;

// print verbose output y/n
int verbose=1;

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
double sig_x = 0.0; // productivity dispersion
double rho_x = 0.0; // productivity persistence
double sig_z = 0.0; // demand dispersion
double rho_z = 0.0; // demand persistence
double corr_z = 0.0; // correlation of productivity shock innovations across destinations
double alpha_n = 0.0; // returns to population size in marketing to new customers
double beta_n = 0.0; // returns to own customer base in marketing to new customers
double gamma_n = 0.0; // diminishing returns to scale in marketing to new customers
double psi_n = 0.0; // marketing efficiency for new customers
double alpha_o = 0.0; // returns to scale in marketing to old customers
double beta_o = 0.0; // returns to own customer base in marketing to old customers
double gamma_o = 0.0; // diminishing returns to scale in marketing to old customers
double psi_o = 0.0; // marketing efficiency for old customers
double z_grid_mult_lb = 0.0;
double z_grid_mult_ub = 0.0;

// customer base grid
double m_grid[NM] = {0.0};

// fixed effect grid
double x_grid[NX] = {0.0}; // grid
double x_hat[NX] = {0.0}; // x^{theta-1} grid
double x_ucond_probs[NX] = {0.0}; // ergodic probabilities
double x_ucond_cumprobs[NX] = {0.0}; // cumulative ergodic probabilities
double x_trans_probs[NX][NX] = {{0.0}}; // transition probabilities
double x_trans_cumprobs[NX][NX] = {{0.0}}; // cumultative transition probabilities
double delta[NX] = {0.0};

// productivity shock grid
double z_grid[NZ] = {0.0}; // grid
double z_hat[NZ] = {0.0}; // z^{theta-1} grid
double z_ucond_probs[NZ] = {0.0}; // ergodic probabilities
double z_ucond_cumprobs[NZ] = {0.0}; // cumulative ergodic probabilities
double z_trans_probs[NZ][NZ] = {{0.0}}; // transition probabilities
double z_trans_cumprobs[NZ][NZ] = {{0.0}}; // cumultative transition probabilities

// destination-specific parameters
char name[ND][3] = {{""}}; // name
double L[ND] = {0.0}; // market size
double Y[ND] = {0.0}; // aggregate consumption index
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
      return La_n[id] * pow(1-m,beta_n) * ( 1 - pow((1.0-m-n)/(1.0-m),1.0-gamma_n) ) / psi_n / (1.0-gamma_n);
      //return pow(L[id],alpha_n) * ( pow(1.0-m,1.0-gamma_n) - pow(1.0-m-n,1.0-gamma_n) ) / psi_n / (1.0-gamma_n);
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
      return La_o[id] * pow(m,beta_o) * ( 1 - pow((m-o)/m,1.0-gamma_o) ) / psi_o / (1.0-gamma_o);
      //return pow(L[id],alpha_o) * ( pow(m,1.0-gamma_o) - pow(m-o,1.0-gamma_o) ) / psi_o / (1.0-gamma_o);
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
      return La_n[id] * pow((1.0-m),beta_n) / psi_n / (1.0-m) / pow((1.0-m-n)/(1.0-m),gamma_n);
      //return pow(L[id],alpha_n) / psi_n  / pow(1.0-m-n,gamma_n);
    }
  else
    {
      return GSL_NAN;
    }
}

/*
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
*/

// derivative of r wrt o
static inline double dr_do(int id, double m, double o)
{
  if(o<m)
    {
      return La_o[id] * pow(m,beta_o) / psi_o / m / pow((m-o)/m,gamma_o);
      //return pow(L[id],alpha_o) / psi_o / pow(m-o,gamma_o);
    }
  else
    {
      return GSL_NAN;
    }
}

/*
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
*/

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
      double x_hi=x_grid_ub_mult*sig_x;
      expspace(x_lo,x_hi,NX,x_grid_exp,x_grid);

      double sum = 0.0;
      for(int i=1; i<NX; i++)
	{
	  x_ucond_probs[i] = pareto_cdf(x_grid[i],sig_x)-pareto_cdf(x_grid[i-1],sig_x);
	  x_ucond_cumprobs[i] = x_ucond_probs[i] +sum;
	  sum += x_ucond_probs[i];
	}
      x_ucond_probs[0] = 1.0 - sum;
    }
  else
    {
      double sum=0.0;
      double m[NX-1];
      for(int i=0; i<NX; i++)
	{
	  if(i<NX-1)
	    m[i] = gsl_cdf_ugaussian_Pinv( ((double)(i+1))/((double)(NX)) ) * sig_x;
	  
	  x_ucond_probs[i] = 1.0/NX;
	  sum += x_ucond_probs[i];

	  if(i==0)
	    x_ucond_cumprobs[i] = x_ucond_probs[i];
	  else
	    x_ucond_cumprobs[i] = x_ucond_cumprobs[i-1] + x_ucond_probs[i];
	}

      if(fabs(sum-1.0)>1.0e-8)
	printf("X probs dont sum to 1!! %0.8f\n",sum);

      x_grid[0] = exp(-sig_x*NX*gsl_ran_gaussian_pdf(m[0]/sig_x,1.0));
      for(int i=1; i<(NX-1); i++)
	{
	  x_grid[i] = exp(-sig_x*NX*(gsl_ran_gaussian_pdf(m[i]/sig_x,1.0)-gsl_ran_gaussian_pdf(m[i-1]/sig_x,1.0)));
	}
      x_grid[NX-1] = exp(sig_x*NX*gsl_ran_gaussian_pdf(m[NX-2]/sig_x,1.0));
    }
  
  for(int i=0; i<NX; i++)
    {
      x_hat[i] = pow(x_grid[i],theta-1.0);
    }

  for(int i=0; i<NX; i++)
    {      
      double sum2=0.0;
      for(int j=0; j<NX; j++)
	{
	  if(i==j)
	    {
	      x_trans_probs[i][j] = rho_x + (1.0-rho_x)*x_ucond_probs[j];
	    }
	  else
	    {
	      x_trans_probs[i][j] = (1.0-rho_x)*x_ucond_probs[j];
	    }
	  
	  sum2 += x_trans_probs[i][j];
	  x_trans_cumprobs[i][j] = sum2;
	}
      
      if(fabs(sum2-1.0)>1.0e-8)
	printf("X trans probs dont sum to 1!! %0.8f\n",sum2);

    }

  /*int n = NX;
  double inprob = 1.0e-8;
  double lo = gsl_cdf_ugaussian_Pinv(inprob)*sig_x*1.5;
  double hi = -gsl_cdf_ugaussian_Pinv(inprob)*sig_x*1.5;
  double ucond_std = sqrt(sig_x*sig_x/(1.0-rho_x*rho_x));
  double d = (hi-lo)/(n-1.0);
  linspace(lo,hi,n,x_grid);
  
  for(int ix=0; ix<n; ix++)
    {
      double x = x_grid[ix];

      double sum=0.0;
      for(int ixp=0; ixp<n; ixp++)
	{
	  double y = x_grid[ixp];
	  
	  x_trans_probs[ix][ixp] = (gsl_cdf_ugaussian_P( (y + d/2.0 - rho_x*x) / sig_x ) -
				       gsl_cdf_ugaussian_P( (y - d/2.0 - rho_x*x) / sig_x ));
	  sum += x_trans_probs[ix][ixp];
	}
      for(int ixp=0; ixp<n; ixp++)
	{
	  x_trans_probs[ix][ixp] = x_trans_probs[ix][ixp]/sum;
	}
    }

  double sum=0.0;
  for(int ix=0; ix<n; ix++)
    {
      double x = x_grid[ix];
      
      x_ucond_probs[ix] = (gsl_cdf_ugaussian_P( (x +  d/2.0) / ucond_std ) -
			  gsl_cdf_ugaussian_P( (x - d/2.0) / ucond_std ));
      sum += x_ucond_probs[ix];
    }
  for(int ix=0; ix<n; ix++)
    {
      x_ucond_probs[ix] = x_ucond_probs[ix]/sum;
    }

  sum=0.0;
  for(int ix=0; ix<n; ix++)
    {
      x_grid[ix] = exp(x_grid[ix]);
      x_hat[ix] = pow(x_grid[ix],theta-1.0);
      sum += x_ucond_probs[ix];
      x_ucond_cumprobs[ix] = sum;

      double sum2=0.0;
      for(int ixp=0; ixp<n; ixp++)
	{
	  sum2 += x_trans_probs[ix][ixp];
	  x_trans_cumprobs[ix][ixp] = sum2;
	}
	}*/  

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

void calc_survival_probs()
{
  for(int ix=0; ix<NX; ix++)
    {
      double death_prob=fmax(0.0,fmin(exp(-delta0*x_hat[ix])+delta1,1.0));
      delta[ix] = 1.0-death_prob;
    }
}

// assigned parameters and initial guesses
int init_params()
{

  /*
andidate parameter vector 44:                                                                           
        sig_x =       1.02000000                                                                         
        rho_x =       0.98194358                                                                         
        sig_z =       0.43933157                                                                         
        rho_z =       0.60418386                                                                         
        z_lb =        3.56360171                                                                         
        z_ub =        2.49261931                                                                         
        psi_n =       0.09784021                                                                         
        alpha_n =     0.50840453                                                                         
        beta_n =      0.93977538                                                                         
        gamma_n =     6.43893845                                                                         
        psi_o =       0.06338877                                                                         
        alpha_o =     0.96266760                                                                         
        beta_o =      0.78924703                                                                         
        gamma_o =     3.82324476                                                                         
        delta0 =      34.65234000                                                                        
        delta1 =      0.00309521    
  */


  // initial guesses for "exporting right away" version
  /*
  W = 1.0;
  Q = 0.86245704;
  delta0 = 34.65234;
  delta1 = 0.00309521;
  theta = 5.0;
  theta_hat = (1.0/theta) * pow(theta/(theta-1.0),1.0-theta);
  //sig_x =  0.90124936;
  sig_x =  1.02;
  rho_x = 0.98194358;
  sig_z = 0.43933157;
  rho_z =  0.60418386;
  alpha_n = 0.50840453;
  alpha_o = 0.96266760;
  beta_n = 0.93977538;
  beta_o = 0.78924476;
  gamma_n = 6.43893845;
  gamma_o = 3.82324476;
  psi_n = 0.09784021;
  psi_o = 0.06338877;
  z_grid_mult_lb=3.56360171;
  z_grid_mult_ub=2.49261931;*/

  W = 1.0;
  Q = 0.86245704;
  delta0 = 20.65234;
  delta1 = 0.015;
  theta = 5.0;
  theta_hat = (1.0/theta) * pow(theta/(theta-1.0),1.0-theta);
  //sig_x =  0.90124936;
  sig_x =  1.05;
  rho_x = 0.98;
  sig_z = 0.35;
  rho_z =  0.25;
  alpha_n = 0.175;
  alpha_o = 0.3;
  beta_n = 0.8;
  beta_o = 0.3;
  gamma_n = 8;
  gamma_o = 5;
  psi_n = 0.09784021;
  psi_o = 0.06338877;
  z_grid_mult_lb=3.5;
  z_grid_mult_ub=3.5;


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
double export_cost2d[ND][NM*NM] = {{0.0}}; // f(m,m') = min{ s(n,m)+r(o,m) s.t. n + o = m' && n in [0,1-m] && o in [0,m] }
double export_cost_argmin_n[ND][NM][NM] = {{{0.0}}}; // argmin[0]
double export_cost_argmin_o[ND][NM][NM] = {{{0.0}}}; // argmin[1]
double export_cost_deriv_m[ND][NM][NM] = {{{0.0}}}; // f_1(m,m') NOTE: grid ordering is [id][im][imp] for interpolation purposes
double export_cost_deriv_mp[ND][NM][NM] = {{{0.0}}}; // f2_m,m') NOTE: grid ordering is [id][im][imp] for interpolation purposes
//double export_cost_deriv_mp_2[ND][NM][NM] = {{{0.0}}}; // f2_m,m') NOTE: grid ordering is [id][imp][im] for interpolation purposes

//gsl_spline * export_cost_spline[ND][NM];
//gsl_interp2d * export_cost_interp2d[ND];

/*
void alloc_cost_spline_mem()
{
  for(int id=0; id<ND; id++)
    {
      //export_cost_interp2d[id] = gsl_interp2d_alloc(gsl_interp2d_bilinear,NM,NM);
      
      for(int im=0; im<NM; im++)
	{
	  export_cost_spline[id][im] = gsl_spline_alloc(gsl_interp_linear,NM);
	}
    }
}

void free_cost_spline_mem()
{
  for(int id=0; id<ND; id++)
    {
      for(int im=0; im<NM; im++)
	{
	  gsl_spline_free(export_cost_spline[id][im]);
	}
      //gsl_interp2d_free(export_cost_interp2d[id]);
    }
}
*/

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
	      export_cost2d[id][imp*NM+im] = s(id,m,mp);
	    }
	  else if(imp==0)
	    {
	      export_cost_argmin_n[id][im][imp]=0.0;
	      export_cost_argmin_o[id][im][imp]=0.0;
	      export_cost[id][im][imp]=0.0;
	      export_cost2d[id][imp*NM+im]=0.0;
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
		  export_cost2d[id][imp*NM+im] = s(id,m,lb+1.0e-11) + r(id,m,mp-lb-1.0e-11);
		}
	      // if marginal cost of aquiring new customers is lower than marginal cost of keeping old ones
	      // even when n=m' and o=0, then corner solution with o=0 is best
	      else if(export_cost_min_foc(ub-1.0e-11,&p) < 0.0)
		{
		  export_cost_argmin_n[id][im][imp]=ub;
		  export_cost_argmin_o[id][im][imp]=mp-ub;
		  export_cost[id][im][imp] = s(id,m,ub-1.0e-11) + r(id,m,mp-ub+1.0e-11);
		  export_cost2d[id][imp*NM+im] = s(id,m,ub-1.0e-11) + r(id,m,mp-ub+1.0e-11);
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
		  export_cost2d[id][imp*NM+im] = s(id,m,export_cost_argmin_n[id][im][imp]) + r(id,m,export_cost_argmin_o[id][im][imp]);
		}
	    }
	}
    }
  

  /*
  for(int im=0; im<NM; im++)
    {
      gsl_spline_init(export_cost_spline[id][im],m_grid,export_cost[id][im],NM);
      }*/
  //gsl_interp2d_init(export_cost_interp2d[id],m_grid,m_grid,export_cost2d[id],NM,NM);
  
  
  for(int im=0; im<NM; im++)
    {
      for(int imp=0; imp<NM; imp++)
	{
	  double dmp=0.0;
	  if(im==0)
	    {
	      dmp = ds_dn(id,0.0,m_grid[imp]);
	    }
	  else if(imp==0)
	    {
	      //dmp = (export_cost[id][im][1]-export_cost[id][im][0])/
	      //		(m_grid[1]-m_grid[0]);

	      
	      if(ds_dn(id,m_grid[im],0.0)<dr_do(id,m_grid[im],0.0))
		{
		  dmp=ds_dn(id,m_grid[im],0.0);
		}
	      else
		{
		  dmp = dr_do(id,m_grid[im],0.0);
		}
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
	  //export_cost_deriv_mp_2[id][imp][im] = dmp;

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

  if(verbose==3)
    printf("\tStatic problem for %.3s completed in %0.0f seconds.\n",name[id],difftime(stop,start));

  return 0;
}

int solve_export_cost_all_dests()
{
  if(verbose)
    printf("Solving static export cost minimization problem for all destinations...\n");
  
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
  WRITE_VECTOR(export_cost_deriv_mp,ND*NM*NM,"output/export_cost_deriv_mp.txt");
  WRITE_VECTOR(export_cost_argmin_n,ND*NM*NM,"output/export_cost_argmin_n.txt");
  WRITE_VECTOR(export_cost_argmin_o,ND*NM*NM,"output/export_cost_argmin_o.txt");
}

///////////////////////////////////////////////////////////////////////////////
// 3. Dynamic program
///////////////////////////////////////////////////////////////////////////////

// equilibrium objects
double gm[ND][NX][NZ][NM] = {{{{0.0}}}}; // policy function for market penetration
double gc[ND][NX][NZ][NM] = {{{{0.0}}}}; // policy function for market penetration
double dV[ND][NX][NZ][NM] = {{{{0.0}}}}; // value function derivative
double EdV_xp[ND][NZ][NM] = {{{0.0}}}; // value function derivative
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
	      //dV[id][ix][iz][im] = 0.0;
	      dV[id][ix][iz][im] = pi_hat[id]*x_hat[ix]*z_hat[iz];
	    }
	}
    }
}

void calc_EdV(int id)
{
  for(int iz=NZ-1; iz>=0; iz--)
    {
      for(int im = NM-1; im>=0; im--)
	{
	  EdV_xp[id][iz][im] = 0.0;
	  for(int ixp; ixp<NX; ixp++)
	    {
	      EdV_xp[id][iz][im] += dV[id][ixp][iz][im]*x_ucond_probs[ixp];
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

	  double Vm2=0.0;
	  if(rho_x<0.999)
	    Vm2 = interp_with_ix(m_grid,EdV_xp[id][izp],NM,mp,imp);
	  
	  EdV += z_trans_probs[iz][izp]*(rho_x*Vm + (1.0-rho_x)*Vm2);
	  //EdV += z_trans_probs[iz][izp]*Vm;
	}
    }

  //return ds_dn(id,0.0,mp) - pi_hat[id]*x_hat[ix]*z_hat[iz] - Q*delta[ix]*EdV;
  return ds_dn(id,0.0,mp) - Q*delta[ix]*EdV;
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
	  //double lower_bound=delta_root;
	  double lower_bound=0.0;
	  double upper_bound = m_grid[NM-1]-delta_root;

	  if(entrant_foc(0.0,&p)>0.0)
	    {
	      gm[id][ix][iz][0] = 0.0;
	      gc[id][ix][iz][0] = 0.0;
	    }
	  else if(entrant_foc(m_grid[NM-1],&p)<0.0)
	    {
	      gm[id][ix][iz][0] = m_grid[NM-1];
	      gc[id][ix][iz][0] = export_cost[id][0][NM-1];
	    }
	  else
	    {
	      /*if(iz>0)
		{
		  lower_bound = fmax(lower_bound,gm[id][ix][iz-1][0]-1000*delta_root);
		}
	      else
		{
		  lower_bound = fmax(lower_bound,gm[id][ix-1][iz][0]-1000*delta_root);
		  }*/
	      gsl_function f;
	      f.function = &entrant_foc;
	      f.params=&p;
	      if(find_root_1d(&f,lower_bound,upper_bound,&(gm[id][ix][iz][0])))
		{
		  printf("\nError solving entrant's first-order condition! (id,ix,iz) = (%d,%d,%d)\n",ix,iz,id);
		  gsl_interp_accel_free(acc);
		  return 1;
		}
	      gc[id][ix][iz][0] = s(id,0,gm[id][ix][iz][0]);
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
	  double Vm = interp_with_ix(m_grid,dV[id][ix][izp],NM,mp,imp);

	  double Vm2 = 0.0;
	  if(rho_x<0.999)
	    Vm2 = interp_with_ix(m_grid,EdV_xp[id][izp],NM,mp,imp);
	  
	  EdV += z_trans_probs[iz][izp]*(rho_x*Vm + (1.0-rho_x)*Vm2);
	  //EdV += z_trans_probs[iz][izp]*Vm;
	}
    }

  double mc = interp_with_ix(m_grid,export_cost_deriv_mp[id][im],NM,mp,imp);
  //double retval = mc - pi_hat[id]*x_hat[ix]*z_hat[iz] - Q*delta[ix]*EdV;
  double retval = mc - Q*delta[ix]*EdV;
  
  if(gsl_isinf(retval) || gsl_isnan(retval))
    {
      printf("Incumbent FOC is INF or NAN!\n");
    }

  return retval;
}

// dV_dm(z,m) = L*Y*(x*z)^{theta-1} - f_1(m,m'(z,m))
double envelope_cond(double mp, void * data)
{
  dp_params * p = (dp_params *)data;
  int ix = p->ix;
  int iz = p->iz;
  int id = p->id;
  int im = p->im;
  gsl_interp_accel * acc = p->acc;

  //double f_1 = -interp(acc,m_grid,export_cost_deriv_m[id][im],NM,mp);
  double f_1 = pi_hat[id]*x_hat[ix]*z_hat[iz] - interp(acc,m_grid,export_cost_deriv_m[id][im],NM,mp);
  
  return f_1;
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
		  gc[id][ix][iz][im] = 0.0;
		}
	      else if(incumbent_foc(m_grid[NM-1],&p)<0.0)
		{
		  gm_tmp = m_grid[NM-1];
		  gc[id][ix][iz][im] = export_cost[id][im][NM-1];
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
	      gc[id][ix][iz][im] = interp(acc,m_grid,export_cost[id][im],NM,gm_tmp);
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
  if(verbose>=3)
    printf("\tSolving policy function for %d = \n",id);
    
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
      calc_EdV(id);

      status = iterate_entrant_policy(id);
      if(status)
	{
	  printf("\tError iterating entrant policy function! id = %d\n",id);
	  break;
	}

      status = iterate_incumbent_policy(id,&maxdiff,imaxdiff);
      if(status)
	{
	  printf("\tError iterating incumbent policy function! id = %d\n",id);
	  break;
	}

      if(verbose==4)
	{
	  printf("\t\tIter %d, diff = %0.2g, loc = (%d, %d, %d), gm[loc] = %0.4g\n",
		 iter,maxdiff,imaxdiff[0],imaxdiff[1],imaxdiff[2],
		 gm[id][imaxdiff[0]][imaxdiff[1]][imaxdiff[2]]);
	}
    }
  while(maxdiff>policy_tol_abs && iter < policy_max_iter);

  time(&stop);

  if(iter==policy_max_iter)
    {
      status=1;
      if(verbose>=3)
	printf("\tPolicy function iteration failed for %.3s! Diff = %0.4g\n",name[id],maxdiff);
    }
  else
    {
      if(verbose>=3)
	{
	  printf("\tPolicy function converged for %.3s in %0.0f seconds!\n",
		 name[id],difftime(stop,start));
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
double x_rand[NF][NT*2];
double z_rand[ND][NF][NT*2];
double surv_rand[NF][NT*2];
double newx_rand[NF][NT*2];
int ix_sim[NF][NT*2];
int iz_sim[ND][NF][NT*2];
double m_sim[ND][NF][NT*2];
double v_sim[ND][NF][NT*2];
double cost_sim[ND][NF][NT*2];
double cost2_sim[ND][NF][NT*2];

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
	  //if(id==0)
	  //  x_rand[i] = gsl_rng_uniform(r);
	  
	  for(int t=0; t<NT*2; t++)
	    {
	      z_rand[id][i][t] = gsl_rng_uniform(r);
	      if(id==0)
		{
		  x_rand[i][t] = gsl_rng_uniform(r);
		  surv_rand[i][t] = gsl_rng_uniform(r);
		  newx_rand[i][t] = gsl_rng_uniform(r);
		}
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
  //if(post_reform)
  //  max_kt = NT*3;

  gsl_interp_accel * acc1 = gsl_interp_accel_alloc();
  gsl_interp_accel * acc2 = gsl_interp_accel_alloc();
  gsl_interp_accel * acc3 = gsl_interp_accel_alloc();

  // then for each firm in the sample...
  for(int jf=0; jf<NF; jf++)
    {
      // find fixed-effect value based on random draw
      gsl_interp_accel_reset(acc1);
      int ix = gsl_interp_accel_find(acc1, x_ucond_cumprobs, NX, x_rand[jf][0]);
      //if(id==0)
      //ix_sim[jf][0] = ix;

      if(ix<0 || ix>=NX)
	{
	  printf("Error!\n");
	}
      
      // find initial value of shock based on random draw and ergodic distribution
      gsl_interp_accel_reset(acc2);
      int iz = gsl_interp_accel_find(acc2, z_ucond_cumprobs, NZ, z_rand[id][jf][0]);
      //iz_sim[id][jf][0] = iz;
      
      // initialize market penetration to zero
      double m = 0.0;
      //m_sim[id][jf][0] = m;
      gsl_interp_accel_reset(acc3);
      
      for(int kt=0; kt<max_kt; kt++)
	{
	  
	  // record current state variables
	  iz_sim[id][jf][kt]=iz;

	  if(id==0)
		ix_sim[jf][kt]=ix;


	  // market penetration using policy function
	  double mp = interp(acc3,m_grid,gm[id][ix][iz],NM,m);
	  m_sim[id][jf][kt] = m;

	  // record current exports (which depend on state variables only!)
	  if(m<1.0e-8)
	    {
	      v_sim[id][jf][kt] = -99.9;
	    }
	  else
	    {
	      v_sim[id][jf][kt] = theta*theta_hat*L[id]*Y[id]*tau_hat[id]*x_hat[ix]*z_hat[iz]*m;
	    }

	  // record exporting costs
	  if(mp<1.0e-8 || m<1.0e-8)
	    {
	      cost_sim[id][jf][kt] = -99.9;
	      cost2_sim[id][jf][kt] = -99.9;
	    }
	  else
	    {
	      double profit = v_sim[id][jf][kt]/theta;
	      double cost=0.0;
	      if(m<1.0e-10)
		{
		  cost=s(id,0,mp);
		}
	      else
		{
		  cost = interp(acc2,m_grid,gc[id][ix][iz],NM,m);
		}
	      
	      cost_sim[id][jf][kt] = cost;
	      cost2_sim[id][jf][kt] = cost/profit;
	    }

	  // unless we are in the very last period of the simulation, update the state variables
	  if(kt<max_kt-1)
	    {
	      // if you die, set market penetration to zero and draw new shocks from ergodic distributions
	      if(surv_rand[jf][kt]>delta[ix])
		{
		  mp = 0.0;
		  ix=gsl_interp_accel_find(acc1, x_ucond_cumprobs, NX, x_rand[jf][kt+1]);
		  iz=gsl_interp_accel_find(acc2, z_ucond_cumprobs, NZ, z_rand[id][jf][kt+1]);
		}
	      // otherwise...
	      else
		{
		  // if you get unlucky, draw a new multilateral productivity
		  if(newx_rand[jf][kt]>rho_x)
		    ix = gsl_interp_accel_find(acc1, x_ucond_cumprobs, NX, x_rand[jf][kt+1]);

		  // draw a new demand shock from conditional distribution
		  iz = gsl_interp_accel_find(acc2, z_trans_cumprobs[iz], NZ, z_rand[id][jf][kt+1]);
		}

	      // regardless, update market penetration
	      m = mp;
	    }	  
	}      
    }

  double expart_rate_[NT] = {0.0};
  double avg_expart_rate=0.0;
  for(int kt=NT; kt<NT*2; kt++)
    {
      for(int jf=0; jf<NF; jf++)
	{
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
  gsl_interp_accel_free(acc3);

  time(&stop);

  if(verbose>=3)
    printf("\tSimulation completed for %.3s in %0.0f seconds. Export part rate = %0.8f.\n",name[id],difftime(stop,start),100*avg_expart_rate);

  return;
}

void simul_all_dests(double * expart_multi,
		     double * exit_multi,
		     double * exit_multi_1d,
		     double * exit_multi_2d,
		     double * exit_multi_3d,
		     double * exit_multi_4d,
		     double * exit_multi_6d,
		     double * exit_multi_10d)
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
  double exit_rate_[NT-1] = {0.0};
  double exit_rate_1d_[NT-1] = {0.0};
  double exit_rate_2d_[NT-1] = {0.0};
  double exit_rate_3d_[NT-1] = {0.0};
  double exit_rate_4d_[NT-1] = {0.0};
  double exit_rate_6d_[NT-1] = {0.0};
  double exit_rate_10d_[NT-1] = {0.0};
  *expart_multi=0.0;
  *exit_multi=0.0;
  *exit_multi_1d=0.0;
  *exit_multi_2d=0.0;
  *exit_multi_3d=0.0;
  *exit_multi_4d=0.0;
  *exit_multi_6d=0.0;
  *exit_multi_10d=0.0;
  int min_kt=NT;
  int max_kt=NT*2;
  for(int kt=min_kt; kt<max_kt; kt++)
    {
      int n1d=0;
      int n2d=0;
      int n3d=0;
      int n4d=0;
      int n6d=0;
      int n10d=0;
      for(int jf=0; jf<NF; jf++)
	{
	  int exporter=0;
	  int exporter_tomorrow=0;
	  int nd=0;
	  
	  for(int id=0; id<ND; id++)
	    {
	      if(v_sim[id][jf][kt]>1.0e-10)
		{
		  nd++;
		  exporter=1;
		}

	      if(kt<max_kt-1 && v_sim[id][jf][kt+1]>1.0e-10)
		{
		  exporter_tomorrow=1;
		}
	    }

	  if(exporter)
	    {
	      expart_rate_[kt-min_kt] += 1.0;
	      
	      if(nd==1)
		n1d++;
	      else if(nd==2)
		n2d++;
	      else if(nd==3)
		n3d++;
	      else if(nd==4)
		n4d++;
	      else if(nd<10)
		n6d++;
	      else if(nd>=10)
		n10d++;		
	    }
	  
	  if(kt<max_kt-1 && exporter==1 && exporter_tomorrow==0)
	    {
	      exit_rate_[kt-min_kt] += 1.0;
	      if(nd==1)
		exit_rate_1d_[kt-min_kt] += 1.0;
	      else if(nd==2)
		exit_rate_2d_[kt-min_kt] += 1.0;
	      else if(nd==3)
		exit_rate_3d_[kt-min_kt] += 1.0;
	      else if(nd==4)
		exit_rate_4d_[kt-min_kt] += 1.0;
	      else if(nd<10)
		exit_rate_6d_[kt-min_kt] += 1.0;
	      else if(nd>=10)
		exit_rate_10d_[kt-min_kt] += 1.0;
	    }
	}

      if(kt<max_kt-1)
	{
	  exit_rate_[kt-min_kt] = exit_rate_[kt-min_kt]/expart_rate_[kt-min_kt];
	  *exit_multi += exit_rate_[kt-min_kt];

	  exit_rate_1d_[kt-min_kt] = exit_rate_1d_[kt-min_kt]/n1d;
	  *exit_multi_1d += exit_rate_1d_[kt-min_kt];

	  if(n2d>0)
	    {
	      exit_rate_2d_[kt-min_kt] = exit_rate_2d_[kt-min_kt]/n2d;
	      *exit_multi_2d += exit_rate_2d_[kt-min_kt];
	    }
	  else
	    {
	      exit_rate_2d_[kt-min_kt] = 0.0;
	      *exit_multi_2d += exit_rate_2d_[kt-min_kt];
	    }

	  if(n3d>0)
	    {
	      exit_rate_3d_[kt-min_kt] = exit_rate_3d_[kt-min_kt]/n3d;
	      *exit_multi_3d += exit_rate_3d_[kt-min_kt];
	    }
	  else
	    {
	      exit_rate_3d_[kt-min_kt] = 0.0;
	      *exit_multi_3d += exit_rate_3d_[kt-min_kt];
	    }

	  if(n4d>0)
	    {
	      exit_rate_4d_[kt-min_kt] = exit_rate_4d_[kt-min_kt]/n4d;
	      *exit_multi_4d += exit_rate_4d_[kt-min_kt];
	    }
	  else
	    {
	      exit_rate_4d_[kt-min_kt] = 0.0;
	      *exit_multi_4d += exit_rate_4d_[kt-min_kt];
	    }

	  if(n6d>0)
	    {
	      exit_rate_6d_[kt-min_kt] = exit_rate_6d_[kt-min_kt]/n6d;
	      *exit_multi_6d += exit_rate_6d_[kt-min_kt];
	    }
	  else
	    {
	      exit_rate_6d_[kt-min_kt] = 0.0;
	      *exit_multi_6d += exit_rate_6d_[kt-min_kt];
	    }

	  
	  if(n10d>0)
	    {
	      exit_rate_10d_[kt-min_kt] = exit_rate_10d_[kt-min_kt]/n10d;
	      *exit_multi_10d += exit_rate_10d_[kt-min_kt];
	    }
	  else
	    {
	      exit_rate_10d_[kt-min_kt] = 0.0;
	      *exit_multi_10d += exit_rate_10d_[kt-min_kt];
	    }
	}

      expart_rate_[kt-min_kt] = expart_rate_[kt-min_kt]/NF;
      *expart_multi += expart_rate_[kt-min_kt];

    }

  *expart_multi=*expart_multi/(max_kt-min_kt);
  *exit_multi=*exit_multi/(max_kt-min_kt-1);
  *exit_multi_1d=*exit_multi_1d/(max_kt-min_kt-1);
  *exit_multi_2d=*exit_multi_2d/(max_kt-min_kt-1);
  *exit_multi_3d=*exit_multi_3d/(max_kt-min_kt-1);
  *exit_multi_4d=*exit_multi_4d/(max_kt-min_kt-1);
  *exit_multi_6d=*exit_multi_6d/(max_kt-min_kt-1);
  *exit_multi_10d=*exit_multi_10d/(max_kt-min_kt-1);
  
  time(&stop);

  if(verbose)
    printf("Finished simulations in %0.0f seconds. Multilateral part. + exit rates = %0.4f, %0.4f.\n",
	   difftime(stop,start),100*(*expart_multi),100*(*exit_multi));

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
  
  fprintf(file,"f,d,y,popt,gdppc,tau,v,m,cost,cost2,ix,iz,nd,nd_group,entry,exit,incumbent,tenure,max_tenure,multilateral_exit\n");
  for(int jf=0; jf<NF; jf++)
    {
      int max_tenure[ND] = {0};

      for(int id=0; id<ND; id++)
	{
	  if(policy_solved_flag[id]==0)
	    {
	      int tenure_=0;
	      for(int kt=min_kt; kt<max_kt; kt++)
		{
		  if(v_sim[id][jf][kt]>1.0e-10)
		    {
		      if(tenure_>max_tenure[id])
			max_tenure[id] = tenure_;

		      tenure_++;
		    }
		  else
		    {
		      tenure_=0;
		    }
		}
	    }
	}

      int tenure[ND] = {0};

      int nd=0;
      int nd_last=0;
      for(int kt=min_kt; kt<max_kt; kt++)
	{
	  nd=0;
	  for(int id=0; id<ND; id++)
	    {
	      if(policy_solved_flag[id]==0 && v_sim[id][jf][kt]>1.0e-10)
		{
		  nd++;
		}
	    }	  
	  if(nd>max_nd)
	    max_nd=nd;

	  int multilateral_exit=0;
	  if(nd==0 && nd_last>0 && kt>0)
	    multilateral_exit=1;

	  nd_last=nd;
	  
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
	      if(policy_solved_flag[id]==0)
		{
		  if(v_sim[id][jf][kt]>1.0e-10)
		    {
		      int exit = (kt<max_kt-1 ? v_sim[id][jf][kt+1]<1.0e-10 : 0);	  
		      int entrant = v_sim[id][jf][kt-1]<0.0;
		      int incumbent = 1-entrant;
		      
		      fprintf(file,"FIRM%d,%.3s,%d,%0.16f,%0.16f,%0.16f,%0.16f,%0.16f,%0.16f,%0.16f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
			      jf,name[id],kt,L[id],Y[id],1.0/tau_hat[id],
			      v_sim[id][jf][kt],m_sim[id][jf][kt],cost_sim[id][jf][kt],cost2_sim[id][jf][kt],
			      ix_sim[jf][kt],iz_sim[id][jf][kt],nd,nd_group,entrant,exit,incumbent,tenure[id],max_tenure[id],multilateral_exit);

		      tenure[id] += 1;
		      max_tenure[id] = (tenure[id]>max_tenure[id] ? tenure[id] : max_tenure[id]);
		    }
		  else
		    {
		      tenure[id] = 0;
		    }
		}	      
	    }
	}
    }

  fclose(file);

  time(&stop);

  if(verbose)
    printf("Panel data construction complete in %0.0f seconds.\n",difftime(stop,start));
}


/////////////////////////////////////////////////////////////////////////////
// 6. Transition dynamics
/////////////////////////////////////////////////////////////////////////////

#ifdef _MODEL_MAIN

const double dist_tol = 1.0e-11;
const int max_dist_iter = 5000;

double dist[ND][NX][NZ][NM] = {{{{0.0}}}};
double tmp_dist[ND][NX][NZ][NM] = {{{{0.0}}}};
double tmp_dist2[ND][NX][NZ][NM] = {{{{0.0}}}};
double tmp_gm[NT][ND][NX][NZ][NM] = {{{{{0.0}}}}};
double tmp_dV[ND][NX][NZ][NM] = {{{{0.0}}}};

int temp_shock_periods = 5;
double tr_tau[ND][NT+1];
double tr_exports[ND][NT+1];
double tr_expart[ND][NT+1];
double tr_mktpen[ND][NT+1];
double tr_te[ND][NT+1];

// initialize distribution
void init_dist(int id)
{
  double sum=0.0;
  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  dist[id][ix][iz][0] = x_ucond_probs[ix] * z_ucond_probs[iz];
	  sum += dist[id][ix][iz][0];
	}
    }
  if(fabs(sum-1.0)>1.0e-8)
    {
      printf("\nInitial distribution does not sum to one! id = %d, sum = %0.4g\n",id,sum);
    }
}

// distribution iteration driver
int update_dist(int id, double new_dist[NX][NZ][NM], double * maxdiff, int *ixs, int *izs, int *ims)
{
  double exit_measure=0.0;
  gsl_interp_accel * acc = gsl_interp_accel_alloc();

  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  for(int im=0; im<NM; im++)
	    {
	      new_dist[ix][iz][im]=0.0;
	    }
	}
    }

  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  gsl_interp_accel_reset(acc);
	  double surv_prob = delta[ix];
	  
	  for(int im=0; im<NM; im++)
	    {
	      exit_measure += dist[id][ix][iz][im]*(1.0-surv_prob);
	      int igm=0;
	      if(gm[id][ix][iz][im]>1.0e-10)
		{
		  igm = gsl_interp_accel_find(acc,m_grid,NM,gm[id][ix][iz][im]);
		}

	      for(int ixp=0; ixp<NX; ixp++)
		{
		  for(int izp=0; izp<NZ; izp++)
		    {
		      new_dist[ix][izp][0] += (1.0-surv_prob)*
			dist[id][ix][iz][im]*x_ucond_probs[ixp]*z_ucond_probs[izp];

		      if(igm==NM-1)
			{
			  new_dist[ix][izp][igm] += dist[id][ix][iz][im]*
			    surv_prob*x_trans_probs[ix][ixp]*z_trans_probs[iz][izp];
			}
		      else if(gm[id][ix][iz][im]<1.0e-10)
			{
			  new_dist[ix][izp][0] += dist[id][ix][iz][im]*
			    surv_prob*x_trans_probs[ix][ixp]*z_trans_probs[iz][izp];
			}
		      else
			{
			  double m1 = (gm[id][ix][iz][im]-m_grid[igm])/
			    (m_grid[igm+1]-m_grid[igm]);
		      
			  double m0 = 1.0-m1;
		      
			  new_dist[ix][izp][igm] += m0*dist[id][ix][iz][im]*
			    surv_prob*x_trans_probs[ix][ixp]*z_trans_probs[iz][izp];
		      
			  new_dist[ix][izp][igm+1] += m1*dist[id][ix][iz][im]*
			    surv_prob*x_trans_probs[ix][ixp]*z_trans_probs[iz][izp]; 
			}
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
	  for(int im=0; im<NM; im++)
	    {
	      sum = sum+new_dist[ix][iz][im];
	      if(fabs(new_dist[ix][iz][im]-dist[id][ix][iz][im])>*maxdiff)
		{
		  *maxdiff = fabs(new_dist[ix][iz][im]-dist[id][ix][iz][im]);
		  *ixs=ix;
		  *izs=iz;
		  *ims=im;
		}
	    }
	}
    }

  if(fabs(sum-1.0)>1.0e-8)
    {
      printf("\nUpdated distribution does not sum to one! id = %d, sum = %0.4g\n",id,sum);
      return 1;
    }

  gsl_interp_accel_free(acc);
  
  return 0;
}

// distribution iteration loop
int stat_dist(int id)
{
  time_t start, stop;
  int iter=0;
  double maxdiff=999;
  int ixs, izs, ims;
  int status=0;

  time(&start);

  init_dist(id);

  do
    {
      iter++;
      status = update_dist(id,tmp_dist[id],&maxdiff,&ixs,&izs,&ims);
      memcpy(dist[id],tmp_dist[id],NX*NZ*NM*sizeof(double));
      
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
      printf("Distribution iteration failed! id = %d, ||H1-H0|| = %0.4g, loc = (%d, %d, %d)\n",id,maxdiff,ixs,izs,ims);
    }
  else if(verbose==3)
    printf("Distribution converged for id = %d, iter = %d, ||H1-H0|| = %0.4g\n",id,iter,maxdiff);

  return status;
}

int stat_dist_all_dests()
{
  if(verbose)
    printf("Solving stationary distributions for all destinations...\n");

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


void calc_tr_dyn(int id, double tr_moments[3])
{
  double expart_rate=0.0;
  double total_exports=0.0;
  double m = 0.0;
  double sumw=0.0;
  
  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  for(int im=0; im<NM; im++)
	    {
	      //if(gm[id][ix][iz][im]>1.0e-8 && dist[id][ix][iz][im]>1.0e-10)
	      if(im>0 && dist[id][ix][iz][im]>1.0e-10)
		{
		  expart_rate += dist[id][ix][iz][im];
		  //m += gm[id][ix][iz][im] * x_hat[ix]*z_hat[iz]* dist[id][ix][iz][im];
		  m += m_grid[im]*x_hat[ix]*z_hat[iz]*dist[id][ix][iz][im];
		  sumw += x_hat[ix]*z_hat[iz]* dist[id][ix][iz][im];
		  double v = theta*theta_hat*L[id]*Y[id]*tau_hat[id]*x_hat[ix]*z_hat[iz]*gm[id][ix][iz][im];
		  total_exports += dist[id][ix][iz][im] * v;
		}
	    }
	}
    }

  //m = m/expart_rate;
  m = m/sumw;

  tr_moments[0] = total_exports;
  tr_moments[1] = expart_rate;
  tr_moments[2] = m;
  
  return;
}

int tr_dyn_perm_tau_chg(int id, double chg)
{
  double tau_hat0 = tau_hat[id];

  memcpy(tmp_dist[id],dist[id],sizeof(double)*NX*NZ*NM);
  memcpy(tmp_gm[0][id],gm[id],sizeof(double)*NX*NZ*NM);
  memcpy(tmp_dV[id],dV[id],sizeof(double)*NX*NZ*NM);
  
  double tr_moments[4];

  // period 0: initial steady state
  calc_tr_dyn(id,tr_moments);
  tr_tau[id][0] = pow(tau_hat[id],1.0/(1.0-theta));
  tr_exports[id][0] = tr_moments[0];
  tr_expart[id][0] = tr_moments[1];
  tr_mktpen[id][0] = tr_moments[2];
  tr_te[id][0] = 0.0;

  // period 1: trade cost changes after firms have made their mkt pen decisions
  tau_hat[id]  = tau_hat0*pow(1.0+chg,1.0-theta);
  calc_tr_dyn(id,tr_moments);  
  tr_tau[id][1] = pow(tau_hat[id],1.0/(1.0-theta));
  tr_exports[id][1] = tr_moments[0];
  tr_expart[id][1] = tr_moments[1];
  tr_mktpen[id][1] = tr_moments[2];
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
      double junk = 0.0;
      int junk2,junk3,junk4;
      if(update_dist(id, tmp_dist2[id], &junk, &junk2, &junk3, &junk4))
	{
	  printf("Error updating distribution!\n");
	  return 1;
	}
      memcpy(dist[id],tmp_dist2[id],NX*NZ*NM*sizeof(double));

      calc_tr_dyn(id,tr_moments);
      tr_tau[id][t+1] = pow(tau_hat[id],1.0/(1.0-theta));
      tr_exports[id][t+1] = tr_moments[0];
      tr_expart[id][t+1] = tr_moments[1];
      tr_mktpen[id][t+1] = tr_moments[2];
      tr_te[id][t+1] = -log(tr_exports[id][t+1]/tr_exports[id][0])/log(1.0+chg);

      /*
      double junk = 0.0;
      int junk2,junk3,junk4;
      if(update_dist(id, tmp_dist2[id], &junk, &junk2, &junk3, &junk4))
	{
	  printf("Error updating distribution!\n");
	  return 1;
	}
	memcpy(dist[id],tmp_dist2[id],NX*NZ*NM*sizeof(double));*/
    }

  // go back to benchmark trade costs, policies, and dist
  memcpy(dist[id],tmp_dist[id],sizeof(double)*NX*NZ*NM);
  memcpy(gm[id],tmp_gm[0][id],sizeof(double)*NX*NZ*NM);
  memcpy(dV[id],tmp_dV[id],sizeof(double)*NX*NZ*NM);
  tau_hat[id] = tau_hat0;
  pi_hat[id] = theta_hat * L[id] * Y[id] * tau_hat[id];

  if(verbose==3)
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

  memcpy(tmp_dist[id],dist[id],sizeof(double)*NX*NZ*NM);
  memcpy(tmp_gm[0][id],gm[id],sizeof(double)*NX*NZ*NM);
  memcpy(tmp_dV[id],dV[id],sizeof(double)*NX*NZ*NM);
  
  double tr_moments[4];

  // period 0: initial steady state
  calc_tr_dyn(id,tr_moments);
  tr_tau[id][0] = pow(tau_hat[id],1.0/(1.0-theta));
  tr_exports[id][0] = tr_moments[0];
  tr_expart[id][0] = tr_moments[1];
  tr_mktpen[id][0] = tr_moments[2];
  tr_te[id][0] = 0.0;

  // period 1: trade cost changes after firms have made their mkt pen decisions
  tau_hat[id]  = tau_hat0*pow(1.0+chg,1.0-theta);
  calc_tr_dyn(id,tr_moments);  
  tr_tau[id][1] = pow(tau_hat[id],1.0/(1.0-theta));
  tr_exports[id][1] = tr_moments[0];
  tr_expart[id][1] = tr_moments[1];
  tr_mktpen[id][1] = tr_moments[2];
  tr_te[id][1] = -log(tr_exports[id][1]/tr_exports[id][0])/log(1.0+chg);

  // period 2 onward: new decision rules
  // first solve new steady state policies with 100% probability of new trade costs 
  pi_hat[id] = theta_hat * L[id] * Y[id] * tau_hat[id];
  if(solve_policies(id))
    {
      printf("Error solving policy function!\n");
      return 1;
    }

  // converge on new policy function with permanent trade policy uncertainty
  int status = 0;
  double maxdiff = 999;
  int imaxdiff[3] = {0};
  
  int iter=0;
  do
    {
      iter++;

      // copy expected continuation value (50% chance of keeping new trade costs, 50% chance of going back)
      for(int ix=0; ix<NX; ix++)
	{
	  for(int iz=0; iz<NZ; iz++)
	    {
	      for(int im=0; im<NM; im++)
		{
		  dV[id][ix][iz][im] = 0.5*dV[id][ix][iz][im] + 0.5*tmp_dV[id][ix][iz][im];
		}
	    }
	}      
      
      calc_EdV(id);
      
      status = iterate_entrant_policy(id);
      if(status)
	{
	  printf("\tError iterating entrant policy function! id = %d\n",id);
	  break;
	}
      
      status = iterate_incumbent_policy(id,&maxdiff,imaxdiff);
      if(status)
	{
	  printf("\tError iterating incumbent policy function! id = %d\n",id);
	  break;
	}
      
      if(verbose==4)
	{
	  printf("\t\tIter %d, diff = %0.2g, loc = (%d, %d, %d), gm[loc] = %0.4g\n",
		 iter,maxdiff,imaxdiff[0],imaxdiff[1],imaxdiff[2],
		 gm[id][imaxdiff[0]][imaxdiff[1]][imaxdiff[2]]);
	}
    }
  while(maxdiff>policy_tol_abs && iter < policy_max_iter);

  if(iter==policy_max_iter)
    {
      status = 1;
      if(verbose>=3)
	printf("\tPolicy function iteration failed for %.3s! Diff = %0.4g\n",name[id],maxdiff);
    }
  else
    {
      if(verbose>=3)
	{
	  printf("\tPolicy function converged for %.3s\n",name[id]);
	}
    }
  
  int t;
  for(t=1; t<NT; t++)
    {
      double junk = 0.0;
      int junk2,junk3,junk4;
      if(update_dist(id, tmp_dist2[id], &junk, &junk2, &junk3, &junk4))
	{
	  printf("Error updating distribution!\n");
	  status = 1;
	}
      memcpy(dist[id],tmp_dist2[id],NX*NZ*NM*sizeof(double));

      calc_tr_dyn(id,tr_moments);
      tr_tau[id][t+1] = pow(tau_hat[id],1.0/(1.0-theta));
      tr_exports[id][t+1] = tr_moments[0];
      tr_expart[id][t+1] = tr_moments[1];
      tr_mktpen[id][t+1] = tr_moments[2];
      tr_te[id][t+1] = -log(tr_exports[id][t+1]/tr_exports[id][0])/log(1.0+chg);
      /*
      double junk = 0.0;
      int junk2,junk3,junk4;
      if(update_dist(id, tmp_dist2[id], &junk, &junk2, &junk3, &junk4))
	{
	  printf("Error updating distribution!\n");
	  status = 1;
	}
	memcpy(dist[id],tmp_dist2[id],NX*NZ*NM*sizeof(double));*/
    }

  // go back to benchmark trade costs, policies, and dist
  memcpy(dist[id],tmp_dist[id],sizeof(double)*NX*NZ*NM);
  memcpy(gm[id],tmp_gm[0][id],sizeof(double)*NX*NZ*NM);
  memcpy(dV[id],tmp_dV[id],sizeof(double)*NX*NZ*NM);
  tau_hat[id] = tau_hat0;
  pi_hat[id] = theta_hat * L[id] * Y[id] * tau_hat[id];

  if(verbose==3)
    printf("\tTransition dynamics complete for id=%id!\n",id);
  
  return status;
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

  memcpy(tmp_dist[id],dist[id],sizeof(double)*NX*NZ*NM);
  memcpy(tmp_gm[0][id],gm[id],sizeof(double)*NX*NZ*NM);
  memcpy(tmp_dV[id],dV[id],sizeof(double)*NX*NZ*NM);
  
  double tr_moments[4];

  // first solve policies backwards
  for(int t=NT-1; t>=1; t--)
    {
      double rer = exp(shock*pow(rho,t));
      tau_hat[id] = tau_hat0 * pow(rer,theta-1.0);
      pi_hat[id] = theta_hat * L[id] * Y[id] * tau_hat[id];
      if(fabs(tau_hat[id]-tau_hat0)>1.0e-3)
	{
	  if(iterate_entrant_policy(id))
	    {
	      printf("\tError iterating entrant policy function! id = %d\n",id);
	      return 1;
	    }

	  double junk1=0;
	  int junk2[3];
	  if(iterate_incumbent_policy(id,&junk1,junk2))
	    {
	      printf("\tError iterating incumbent policy function! id = %d\n",id);
	      return 1;
	    }
	  memcpy(tmp_gm[t][id],gm[id],sizeof(double)*NX*NZ*NM);
	}
      else
	{
	  memcpy(tmp_gm[t][id],tmp_gm[0][id],sizeof(double)*NX*NZ*NM);
	}
    }

  // now iterate forward

  // period 0: initial steady state
  memcpy(gm[id],tmp_gm[0][id],sizeof(double)*NX*NZ*NM);
  tau_hat[id] = tau_hat0;
  calc_tr_dyn(id,tr_moments);
  tr_tau[id][0] = pow(tau_hat[id],1.0/(1.0-theta));
  tr_exports[id][0] = tr_moments[0];
  tr_expart[id][0] = tr_moments[1];
  tr_mktpen[id][0] = tr_moments[2];
  tr_te[id][0] = 0.0;

  // period 1: trade cost changes after firms have made their mkt pen decisions
  tau_hat[id] = tau_hat0 * pow(exp(shock),theta-1.0);
  calc_tr_dyn(id,tr_moments);  
  tr_tau[id][1] = pow(tau_hat[id],1.0/(1.0-theta));
  tr_exports[id][1] = tr_moments[0];
  tr_expart[id][1] = tr_moments[1];
  tr_mktpen[id][1] = tr_moments[2];
  tr_te[id][1] = -log(tr_exports[id][1]/tr_exports[id][0])/(-log(1.0+shock));

  // period 2 onward: new decision rules  
  int t;
  for(t=1; t<NT; t++)
    {
      memcpy(gm[id],tmp_gm[t][id],NX*NZ*NM*sizeof(double));
      
      double junk = 0.0;
      int junk2,junk3,junk4;
      if(update_dist(id, tmp_dist2[id], &junk, &junk2, &junk3, &junk4))
	{
	  printf("Error updating distribution!\n");
	  return 1;
	}
      memcpy(dist[id],tmp_dist2[id],NX*NZ*NM*sizeof(double));

      double rer = exp(shock*pow(rho,t));
      tau_hat[id] = tau_hat0 * pow(rer,theta-1.0);
      pi_hat[id] = theta_hat * L[id] * Y[id] * tau_hat[id];
      calc_tr_dyn(id,tr_moments);
      tr_tau[id][t+1] = pow(tau_hat[id],1.0/(1.0-theta));
      tr_exports[id][t+1] = tr_moments[0];
      tr_expart[id][t+1] = tr_moments[1];
      tr_mktpen[id][t+1] = tr_moments[2];
      tr_te[id][t+1] = -log(tr_exports[id][t+1]/tr_exports[id][0])/(-log(rer));
      /*
      double junk = 0.0;
      int junk2,junk3,junk4;
      if(update_dist(id, tmp_dist2[id], &junk, &junk2, &junk3, &junk4))
	{
	  printf("Error updating distribution!\n");
	  return 1;
	}
	memcpy(dist[id],tmp_dist2[id],NX*NZ*NM*sizeof(double));*/
    }

  // go back to benchmark trade costs, policies, and dist
  memcpy(dist[id],tmp_dist[id],sizeof(double)*NX*NZ*NM);
  memcpy(gm[id],tmp_gm[0][id],sizeof(double)*NX*NZ*NM);
  memcpy(dV[id],tmp_dV[id],sizeof(double)*NX*NZ*NM);
  tau_hat[id] = tau_hat0;
  pi_hat[id] = theta_hat * L[id] * Y[id] * tau_hat[id];

  if(verbose==3)
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
      fprintf(file,"d,popt,gdppc,t,tau,exports,expart_rate,mktpen_rate,trade_elasticity\n");
      for(int id=0; id<ND; id++)
	{
	  if(policy_solved_flag[id]==0)
	    {
	      for(int it=0; it<NT+1; it++)
		{
		  fprintf(file,"%.3s,%0.16f,%0.16f,%d,%0.16f,%0.16f,%0.16f,%0.16f,%0.16f\n",
			  name[id],L[id],Y[id],it,tr_tau[id][it],tr_exports[id][it],
			  tr_expart[id][it],tr_mktpen[id][it],tr_te[id][it]);
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
// 7. Life cycle dynamics
//////////////////////////////////////////////////////////////////////////////
double lf_m[ND][NX][NZ][NT] = {{{{0.0}}}};
double lf_fcost[ND][NX][NZ][NT] = {{{{0.0}}}};
double lf_fcost2[ND][NX][NZ][NT] = {{{{0.0}}}};

int calc_lf_dyn(int id)
{
  gsl_interp_accel * acc = gsl_interp_accel_alloc();
  
  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  if(gm[id][ix][iz][0]>1.0e-10)
	    {
	      gsl_interp_accel_reset(acc);
	      double m = 0.0;
	      for(int it=0; it<NT; it++)
		{
		  double mp = interp(acc,m_grid,gm[id][ix][iz],NM,m);
		  double profit = theta_hat*mp*L[id]*Y[id]*tau_hat[id]*x_hat[ix]*z_hat[iz];
		  double cost=0.0;
		  if(m<1.0e-10)
		    {
		      cost=s(id,0.0,mp);
		    }
		  else
		    {
		      cost = interp(acc,m_grid,gc[id][ix][iz],NM,m);
		    }
		  //cost = interp2(acc,m_grid,NM,export_cost[id],m,mp);
		  
		  lf_m[id][ix][iz][it] = mp;
		  lf_fcost[id][ix][iz][it] = cost;
		  lf_fcost2[id][ix][iz][it] = cost/profit;
		  m=mp;
		}
	    }
	}
    }

  gsl_interp_accel_free(acc);

  if(verbose==3)
    printf("\tDestination %d of %d complete!\n",id,ND);
  
  return 0;

}

int calc_lf_dyn_all_dests()
{
  printf("Computing life cycle profiles...\n");

  time_t start, stop;
  time(&start);
	       
  int error=0;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int id=0; id<ND; id++)
    {
      if(policy_solved_flag[id]==0)
	{
	  if(calc_lf_dyn(id))
	    error=1;
	}
    }
  
  time(&stop);
  printf("Complete! Time: %0.0f seconds.\n",difftime(stop,start));

  return error;
}


int write_lf_dyn_results(char * fname)
{
  FILE * file = fopen(fname,"w");
  if(file)
    {
      fprintf(file,"d,popt,gdppc,tau,ix,iz,t,mktpen,cost,cost_profit_ratio\n");
      for(int id=0; id<ND; id++)
	{
	  for(int ix=0; ix<NX; ix++)
	    {
	      for(int iz=0; iz<NZ; iz++)
		{
		  if(gm[id][ix][iz][0]>1.0e-10)
		    {
		      for(int it=0; it<NT; it++)
			{
			  fprintf(file,"%.3s,%0.16f,%0.16f,%0.16f,%d,%d,%d,%0.16f,%0.16f,%0.16f\n",
				  name[id],L[id],Y[id],tau_hat[id],
				  ix,iz,it,lf_m[id][ix][iz][it],
				  lf_fcost[id][ix][iz][it],lf_fcost2[id][ix][iz][it]);
			}
		    }
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
// 8. Main work function
///////////////////////////////////////////////////////////////////////////////

#define NP 16
#define NY 82
FILE * results_file;
int fcnt=0;

//int work(const double params[NP], double residuals[NY])
int work(const double params[NP], double * error)
{
  linebreak();

  printf("Solving and simulating model...\n");

  time_t start, stop;
  time(&start);

  fcnt++;

  sig_x = params[0];
  rho_x = params[1];
  sig_z = params[2];
  rho_z = params[3];
  psi_n = params[4];
  alpha_n = params[5];
  gamma_n = params[6];
  psi_o = params[7];
  alpha_o = params[8];
  gamma_o = params[9];
  delta0 = params[10];
  delta1 = params[11];
  beta_n = params[12];
  beta_o = params[13];
  z_grid_mult_lb = params[14];
  z_grid_mult_ub = params[15];

  if(results_file)
    {
      for(int i = 0; i<NP; i++)
	{
	  fprintf(results_file,"%0.16f,",params[i]);
	}
    }
      
  printf("\nCandidate parameter vector %d:\n",fcnt);
  printf("\tsig_x =       %0.8f\n",sig_x);
  printf("\trho_x =       %0.8f\n",rho_x);
  printf("\tsig_z =       %0.8f\n",sig_z);
  printf("\trho_z =       %0.8f\n",rho_z);
  printf("\tz_lb =        %0.8f\n",z_grid_mult_lb);
  printf("\tz_ub =        %0.8f\n",z_grid_mult_ub);
  printf("\tpsi_n =       %0.8f\n",psi_n);
  printf("\talpha_n =     %0.8f\n",alpha_n);
  printf("\tbeta_n =      %0.8f\n",beta_n);
  printf("\tgamma_n =     %0.8f\n",gamma_n);
  printf("\tpsi_o =       %0.8f\n",psi_o);
  printf("\talpha_o =     %0.8f\n",alpha_o);
  printf("\tbeta_o =      %0.8f\n",beta_o);
  printf("\tgamma_o =     %0.8f\n",gamma_o);
  printf("\tdelta0 =      %0.8f\n",delta0);
  printf("\tdelta1 =      %0.8f\n\n",delta1);
  
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
      return 1;
  save_export_cost();
    
  if(solve_policies_all_dests())
    return 1;
  
  double expart_multi=0.0;
  double exit_multi=0.0;
  double exit_multi_1d=0.0;
  double exit_multi_2d=0.0;
  double exit_multi_3d=0.0;
  double exit_multi_4d=0.0;
  double exit_multi_6d=0.0;
  double exit_multi_10d=0.0;
  simul_all_dests(&expart_multi,&exit_multi,&exit_multi_1d,&exit_multi_2d,&exit_multi_3d,&exit_multi_4d,&exit_multi_6d,&exit_multi_10d);
  create_panel_dataset("output/model_microdata_calibration.csv");

  time(&stop);
  printf("\nCalling python scripts to process and analyze simulated microdata...\n");
  
  if(system("python3 -W ignore ../python/model_microdata_prep.py dmp"))
    return 1;
  
  if(system("python3 -W ignore ../python/sumstats_regs.py"))
    return 1;

  if(system("python3 -W ignore ../python/life_cycle.py"))
  return 1;

  printf("\nData processing complete! Time: %0.0f seconds.\n",difftime(stop,start));

  FILE * file = fopen("../python/output/calibration_data2.txt","r");
  if(!file)
    {
      printf("Failed to open file with calibration data!\n");
      return 1;
    }
  else
    {
      double data_moments[NY];
      double model_moments[NY];
      double weights[NY];
      double residuals[NY];
      
      int got = 0;
      for(int i=0; i<NY-8; i++)
	{
	  got += fscanf(file,"%lf",&(data_moments[i]));
	}
      for(int i=0; i<NY-8; i++)
	{
	  got += fscanf(file,"%lf",&(model_moments[i]));
	}
      for(int i=0; i<NY-8; i++)
	{
	  got += fscanf(file,"%lf",&(weights[i]));
	}
	    
      fclose(file);
      if(got!=3*(NY-8))
	{
	  printf("Failed to load calibration data! Got = %d\n",got);
	  return 1;
	}
      else
	{
	  
	  data_moments[NY-8] = 0.25;
	  model_moments[NY-8] = expart_multi;
	  
	  data_moments[NY-7] = 0.34;
	  model_moments[NY-7] = exit_multi;

	  
	  data_moments[NY-6] = 0.566251;
	  model_moments[NY-6] = exit_multi_1d;

	  data_moments[NY-5] = 0.375161;
	  model_moments[NY-5] = exit_multi_2d;

	  data_moments[NY-4] = 0.245397;
	  model_moments[NY-4] = exit_multi_3d;

	  data_moments[NY-3] = 0.164864;
	  model_moments[NY-3] = exit_multi_4d;

	  data_moments[NY-2] = 0.08765;
	  model_moments[NY-2] = exit_multi_6d;

	  data_moments[NY-1] = 0.023675;
	  model_moments[NY-1] = exit_multi_10d;

	  weights[NY-8] = 0.1;
	  weights[NY-7] = 0.008915307195321662;
	  weights[NY-6] = 0.006238;
	  weights[NY-5] = 0.006970;
	  weights[NY-4] = 0.005813;
	  weights[NY-3] = 0.004722;
	  weights[NY-2] = 0.003073;
	  weights[NY-1] = 0.001910;

	  *error = 0.0;
	  double sum = 0.0;
	  for(int i=0; i<NY; i++)
	    {
	      double tmp = fabs(data_moments[i]-model_moments[i]);
	      //	      if(i==3 || i==4 || i==17 || i==18 || i==19)
	      //tmp = tmp/fabs(data_moments[i]);
	      
	      *error += tmp*tmp/(weights[i]*weights[i]);
	      ////*error += tmp*tmp/(weights[i]*weights[i]);
	      sum += 1.0/(weights[i]*weights[i]);
	      
	      residuals[i] = tmp;
	      
	      if(results_file)
		{
		  fprintf(results_file,"%0.16f",tmp);
		  if(i<NY-1)
		    fprintf(results_file,",");
		}
	    }

	  if(results_file)
	    {
	      fprintf(results_file,"\n");
	      fflush(results_file);
	    }
	  //*error = sqrt(*error/NY);
	  *error = sqrt(*error/sum);

	  int fails=0;
	  for(int j=0; j<ND; j++)
	    {
	      fails += policy_solved_flag[j];
	    }
	  *error = *error + (double)(fails)/((double)(ND));

	  time(&stop);

	  if(verbose>=2)
	    {
	      printf("-------------------------------------------------\n");
	      
	      printf("\nMoments:                               data model diff:\n\n");

	      printf("\tNum. exporters (cv):           %+0.4f %+0.4f %0.4f\n",
		     data_moments[0],model_moments[0],residuals[0]);
	      printf("\tTop 5 share (avg):             %+0.4f %+0.4f %0.4f\n",
		     data_moments[1],model_moments[1],residuals[1]);
	      //printf("\tTop 5 share (iqr):             %+0.4f %+0.4f %0.4f\n",
	      //	     data_moments[2],model_moments[2],residuals[2]);
	      printf("\tAvg num dest (avg):            %+0.3f %+0.3f %0.4f\n",
		     data_moments[2],model_moments[2],residuals[2]);
	      //printf("\tAvg num dest (iqr):            %+0.4f %+0.4f %0.4f\n",
	      //	     data_moments[4],model_moments[4],residuals[4]);
	      printf("\tExit rate (avg):               %+0.4f %+0.4f %0.4f\n",
		     data_moments[3],model_moments[3],residuals[3]);
	      //printf("\tExit rate (iqr):               %+0.4f %+0.4f %0.4f\n",
	      //     data_moments[6],model_moments[6],residuals[6]);
	      printf("\tEntrant rel size (avg):        %+0.4f %+0.4f %0.4f\n",
		     data_moments[4],model_moments[4],residuals[4]);
	      //printf("\tEntrant rel size (iqr):        %+0.4f %+0.4f %0.4f\n",
	      //	     data_moments[8],model_moments[8],residuals[8]);
	      printf("\tEntrant rel exit rate (avg):   %+0.4f %+0.4f %0.4f\n",
		     data_moments[5],model_moments[5],residuals[5]);
	      //printf("\tEntrant rel exit rate (iqr):   %+0.4f %+0.4f %0.4f\n",
	      //     data_moments[10],model_moments[10],residuals[10]);

	      int cnt=6;
	      
	      printf("\tNum. exporters (beta rgdp):    %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt],model_moments[cnt],residuals[cnt]);
	      printf("\tNum. exporters (beta pop):     %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt+1],model_moments[cnt+1],residuals[cnt+1]);
	      printf("\tNum. exporters (beta tau):     %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt+2],model_moments[cnt+2],residuals[cnt+2]);
	      cnt=cnt+3;

	      printf("\tTop 5 share (beta rgdp):       %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt],model_moments[cnt],residuals[cnt]);
	      printf("\tTop 5 share (beta pop):        %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt+1],model_moments[cnt+1],residuals[cnt+1]);
	      printf("\tTop 5 share (beta tau):        %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt+2],model_moments[cnt+2],residuals[cnt+2]);
	      cnt=cnt+3;

	      printf("\tAvg num dest (beta rgdp):      %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt],model_moments[cnt],residuals[cnt]);
	      printf("\tAvg num dest (beta pop):       %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt+1],model_moments[cnt+1],residuals[cnt+1]);
	      printf("\tAvg num dest (beta tau):       %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt+2],model_moments[cnt+2],residuals[cnt+2]);
	      cnt=cnt+3;

	      printf("\tExit rate (beta rgdp):         %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt],model_moments[cnt],residuals[cnt]);
	      printf("\tExit rate (beta pop):          %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt+1],model_moments[cnt+1],residuals[cnt+1]);
	      printf("\tExit rate (beta tau):          %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt+2],model_moments[cnt+2],residuals[cnt+2]);
	      cnt=cnt+3;

	      printf("\tEntrant rel size (beta rgdp):  %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt],model_moments[cnt],residuals[cnt]);
	      printf("\tEntrant rel size (beta pop):   %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt+1],model_moments[cnt+1],residuals[cnt+1]);
	      printf("\tEntrant rel size (beta tau):   %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt+2],model_moments[cnt+2],residuals[cnt+2]);
	      cnt=cnt+3;

	      printf("\tEntrant rel exit (beta rgdp):  %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt],model_moments[cnt],residuals[cnt]);
	      printf("\tEntrant rel exit (beta pop):   %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt+1],model_moments[cnt+1],residuals[cnt+1]);
	      printf("\tEntrant rel exit (beta tau):   %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt+2],model_moments[cnt+2],residuals[cnt+2]);
	      cnt=cnt+3;

	      printf("\n\tEffects of tenure on survival (hard)\n");
	      printf("\t1:                            %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt],model_moments[cnt],residuals[cnt]);
	      printf("\t2:                            %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt+1],model_moments[cnt+1],residuals[cnt+1]);
	      printf("\t3:                            %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt+2],model_moments[cnt+2],residuals[cnt+2]);
	      printf("\t4:                            %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt+3],model_moments[cnt+3],residuals[cnt+3]);
	      printf("\t5:                            %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt+4],model_moments[cnt+4],residuals[cnt+4]);
	      cnt = cnt+5;

	      printf("\n\tEffects of tenure on survival (easy)\n");
	      printf("\t1:                            %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt],model_moments[cnt],residuals[cnt]);
	      printf("\t2:                            %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt+1],model_moments[cnt+1],residuals[cnt+1]);
	      printf("\t3:                            %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt+2],model_moments[cnt+2],residuals[cnt+2]);
	      printf("\t4:                            %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt+3],model_moments[cnt+3],residuals[cnt+3]);
	      printf("\t5:                            %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt+4],model_moments[cnt+4],residuals[cnt+4]);
	      cnt=cnt+5;

	      printf("\n\tEffects of tenure:duration on sales (hard)\n");
	      for(int duration=1; duration<=5; duration++)
		{
		 for(int tenure=0; tenure<=duration; tenure++)
		   {
		    printf("\t%d:%d                            %+0.4f %+0.4f %0.4f\n",
			   tenure,duration,data_moments[cnt],model_moments[cnt],residuals[cnt]);
		    cnt++;
		   }
		}

	      printf("\n\tEffects of tenure:duration on sales (easy)\n");
	      for(int duration=1; duration<=5; duration++)
		{
		 for(int tenure=0; tenure<=duration; tenure++)
		   {
		    printf("\t%d:%d                            %+0.4f %+0.4f %0.4f\n",
			   tenure,duration,data_moments[cnt],model_moments[cnt],residuals[cnt]);
		    cnt++;
		   }
		}

	      
	      printf("\n\tMultilateral ex. part. rate:   %0.4f %0.4f %0.4f\n",
		     data_moments[cnt],model_moments[cnt],residuals[cnt]);
	      printf("\tMultilateral exit rate:        %0.4f %0.4f %0.4f\n",
		     data_moments[cnt+1],model_moments[cnt+1],residuals[cnt+1]);
	      printf("\tMultilateral exit rate (1D):   %0.4f %0.4f %0.4f\n",
		     data_moments[cnt+2],model_moments[cnt+2],residuals[cnt+2]);
	      printf("\tMultilateral exit rate (2D):   %0.4f %0.4f %0.4f\n",
		     data_moments[cnt+3],model_moments[cnt+3],residuals[cnt+3]);
	      printf("\tMultilateral exit rate (3D):   %0.4f %0.4f %0.4f\n",
		     data_moments[cnt+4],model_moments[cnt+4],residuals[cnt+4]);
	      printf("\tMultilateral exit rate (4D):   %0.4f %0.4f %0.4f\n",
		     data_moments[cnt+5],model_moments[cnt+5],residuals[cnt+5]);
	      printf("\tMultilateral exit rate (6D):   %0.4f %0.4f %0.4f\n",
		     data_moments[cnt+6],model_moments[cnt+6],residuals[cnt+6]);
	      printf("\tMultilateral exit rate (10D):  %0.4f %0.4f %0.4f\n",
		     data_moments[cnt+7],model_moments[cnt+7],residuals[cnt+7]);
	      
	      printf("\n\tNumber of failed destinations:       %d\n",fails);
	      /*printf("\tMultilateral ex. part. rate:   %+0.4f %+0.4f %0.4f\n",
		     data_moments[cnt],model_moments[cnt],residuals[cnt]);
	      printf("\tMultilateral exit rate:        %+0.4f %+0.4f %0.4f\n",
	      data_moments[cnt+1],model_moments[cnt+1],residuals[cnt+1]);*/
	    }
	  
	  printf("\nFitness evaluation complete! Runtime = %0.0f seconds. Error = %0.8f\n",difftime(stop,start),*error);

	  
	  return 0;
	}
    }
}

//int work_gsl_wrapper(const gsl_vector * x, void * data, gsl_vector * f)
double work_nlopt_wrapper(unsigned n, const double *x, double *grad, void *my_func_data)
{
  nlopt_opt opt = (nlopt_opt)my_func_data;
  double error=0;
      
  if(n != NP)
    {
      nlopt_force_stop(opt);
    }

  if(work(x,&error))
    {
      nlopt_force_stop(opt);
    }      

  return error;
}

///////////////////////////////////////////////////////////////////////////////
// 9. Main function and wrappers for non-calibration exercises
///////////////////////////////////////////////////////////////////////////////

int setup()
{
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
  calc_survival_probs();
  discretize_m();
  random_draws();
  
  return 0;
}

int main(int argc, char * argv[])
{
  int calibrate=0;
  int transitions=0;
  if(argc>2)
    {
      return 1;
    }
  else if(argc==2 && strcmp(argv[1],"-calibrate")==0)
    {
      calibrate=1;
    }
  else if(argc==2 && strcmp(argv[1],"-transitions")==0)
    {
      transitions=1;
    }  
  
  time_t start, stop;
  time(&start);

  // setup environment
  linebreak();
  linebreak();
  if(setup())
      return 1;

  double params[NP];
  double residuals[NF];

  params[0] = sig_x;
  params[1] = rho_x;
  params[2] = sig_z;
  params[3] = rho_z;
  params[4] = psi_n;
  params[5] = alpha_n;
  params[6] = gamma_n;
  params[7] = psi_o;
  params[8] = alpha_o;
  params[9] = gamma_o;
  params[10] = delta0;
  params[11] = delta1;
  params[12] = beta_n;
  params[13] = beta_o;
  params[14] = z_grid_mult_lb;
  params[15] = z_grid_mult_ub;

  // solve model once
  if(calibrate==0)
    {
      verbose=2;
      linebreak();	  
      if(work(params,residuals))
	return 1;
    }
  //calibrate parameters
  else if(calibrate==1)
    {
      verbose=2;
      linebreak();
      linebreak();
      printf("Calibrating model parameters...\n");
      linebreak();
      linebreak();


      // sig_x, rho_x
      // sig_z, rho_z
      // psi_n, alpha_n, gamma_n
      // psi_o, alpha_o, gamma_o
      // delta1, delta0
      // beta_n, beta_o
      // zlo, zhi
        
      double lb[NP] = {0.95, 0.8,
		       0.25, 0.6,
		       0.04, 0.2, 4.0,
		       0.04, 0.25, 2.25,
		       25.0, 0.0005,
		       0.2, 0.2,
		       2.0,2.0};
      double ub[NP] = {1.15, 0.99,
		       0.5, 0.925,
		       0.125, 1.3, 7.5,
		       0.125, 1.3, 5.0,
		       45.0, 0.01,
		       1.3, 1.3,
		       4.0,4.0};
  
      nlopt_opt opt;

      opt = nlopt_create(NLOPT_LN_SBPLX, NP);
      nlopt_set_lower_bounds(opt, lb);
      nlopt_set_upper_bounds(opt, ub);
      nlopt_set_min_objective(opt, work_nlopt_wrapper, opt);
      nlopt_set_xtol_rel(opt, 1e-4);
      nlopt_set_maxeval(opt, 3000);
      
      double minf; /* `*`the` `minimum` `objective` `value,` `upon` `return`*` */
      results_file = fopen("output/results.csv","a");
      int status = nlopt_optimize(opt, params, &minf);
      if(status<0)
	{
	  linebreak();
	  linebreak();
	  printf("nlopt failed with code = %d!\n",status);
	  linebreak();
	  linebreak();
	}
      else
	{
	  linebreak();
	  linebreak();
	  printf("nlopt found minimum!");
	  verbose=2;
	  work(params,&minf);
	  linebreak();
	  linebreak();
	}

      fclose(results_file);
      nlopt_destroy(opt);
    }

  // do transition dynamics
  if(transitions)
    {
      // life cycle profiles
      linebreak();
      if(calc_lf_dyn_all_dests())
	return 1;
      if(write_lf_dyn_results("output/lf_dyn.csv"))
	return 1;

      // solve stationary distributions
      linebreak();
      if(stat_dist_all_dests())
	return 1;

      // effects of permanent drop in trade costs
      linebreak();  
      if(tr_dyn_perm_tau_chg_all_dests(-0.1))
	return 1;
      if(write_tr_dyn_results("output/tr_dyn_perm_tau_drop.csv"))
	return 1;

      // effects of permanent drop in trade costs with uncertainty
      linebreak();  
      if(tr_dyn_perm_tau_chg_uncertain_all_dests(-0.1))
	return 1;
      if(write_tr_dyn_results("output/tr_dyn_perm_tau_drop_uncertain.csv"))
	return 1;

      // effects of temporary good depreciation
      double shock = log(1.0+(theta-1.0)/10.0)/(theta-1.0);
      linebreak();
      if(tr_dyn_rer_shock_all_dests(shock,0.9))
	return 1;
      if(write_tr_dyn_results("output/tr_dyn_rer_dep.csv"))
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

