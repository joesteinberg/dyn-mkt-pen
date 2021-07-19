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
#define NP 16
#define NY 82 //= (6+18) + (10+40) + 8
//#define NY 64 //= (3+10) + (10+40) + 1

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
  printf("\n////////////////////////////////////////////////////////////////////////////\n");
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
double alpha2_n = 0.0; // returns to population size in marketing to new customers
double beta_n = 0.0; // returns to own customer base in marketing to new customers
double gamma_n = 0.0; // diminishing returns to scale in marketing to new customers
double psi_n = 0.0; // marketing efficiency for new customers
double alpha_o = 0.0; // returns to scale in marketing to old customers
double alpha2_o = 0.0; // returns to scale in marketing to old customers
double beta_o = 0.0; // returns to own customer base in marketing to old customers
double gamma_o = 0.0; // diminishing returns to scale in marketing to old customers
double psi_o = 0.0; // marketing efficiency for old customers
double z_grid_mult_lb = 0.0;
double z_grid_mult_ub = 0.0;
double phi0 = 0.0;
double phi1 = 0.0;
double om0 = 0.0;
double om1 = 0.0;
double chi = 0.5;
double zeta = 0.0;
 
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
double ta_n[ND] = {0.0}; // = t^(alpha_n)
double tam_n[ND] = {0.0}; // = t^(alpha_n-1)
double ta_o[ND] = {0.0}; // = t^(alpha_o)
double tam_o[ND] = {0.0}; // = t^(alpha_o-1)
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
      double tmp1 = La_n[id] * pow((1.0-m),beta_n) / psi_n / (1.0-gamma_n);
      double d_tmp1 = -beta_n * pow(1.0-m,beta_n-1.0) * La_n[id] / psi_n / (1.0-gamma_n);
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
      double tmp1 = La_i[id] * pow(m, beta_o) / psi_o / (1.0-gamma_o);
      double d_tmp1 = beta_o * pow(m,beta_o-1.0) * La_o[id] / psi_o / (1.0-gamma_o);
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

  W = 1.0;
  Q = 0.86245704;
  theta = 5.0;
  theta_hat = (1.0/theta) * pow(theta/(theta-1.0),1.0-theta);
  
  // version where we calibrate to means from sum stats and coefficients on dest. characteristics
  delta0 = 34.65234;
  delta1 = 0.00309521;
  sig_x =  1.02;
  rho_x = 0.98194358;
  sig_z = 0.43933157;
  rho_z =  0.60418386;
  alpha_n = 0.50840453;
  alpha_o = 0.96266760;
  alpha2_n = 0.0;
  alpha2_o = 0.0;
  beta_n = 0.94;
  beta_o = 0.78924476;
  gamma_n = 6.43893845;
  gamma_o = 3.82324476;
  psi_n = 0.09784021;
  psi_o = 0.06338877;
  //psi_n = psi_n/1.5;
  //psi_o = psi_o/1.5;
  z_grid_mult_lb=3.56360171;
  z_grid_mult_ub=2.49261931;
  phi1 = HUGE_VAL;
  om1 = 0.0;
  chi = HUGE_VAL;
  zeta = 0;

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
	      ta_n[id] = pow(tau_hat[id],alpha2_n);
	      tam_n[id] = pow(tau_hat[id],alpha2_n-1.0);
	      ta_o[id] = pow(tau_hat[id],alpha2_o);
	      tam_o[id] = pow(tau_hat[id],alpha2_o-1.0);
	      pi_hat[id] = theta_hat * L[id] * Y[id] * tau_hat[id];
	    }
	}

      return 0;
    }
}

///////////////////////////////////////////////////////////////////////////////
// 3. Dynamic program
///////////////////////////////////////////////////////////////////////////////

// equilibrium objects
double m_grid[NM];
double gn[ND][NX][NZ][NM] = {{{{0.0}}}}; // policy function for acquisition
double gm[ND][NX][NZ][NM] = {{{{0.0}}}}; // policy function for retention
double gs[ND][NX][NZ][NM] = {{{{0.0}}}}; // policy function for acquisition cost
double gr[ND][NX][NZ][NM] = {{{{0.0}}}}; // policy function for retention cost
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

double acquisition_foc(double n, void * data)
{
  dp_params * p = (dp_params *)data;
  int id = p->id;
  int ix = p->ix;
  int iz = p->iz;
  int im = p->im;
  gsl_interp_accel * acc = p->acc;

  double m = m_grid[im];
  double m2 = m+n;
  int im2 = gsl_interp_accel_find(acc, m_grid, NM, m2);
  double m3_interp = interp_with_ix(m_grid,gm[id][ix][iz],NM,m2,im2);
  double r1 = dr_dm(m2,m3);
  double s2 = ds_dn(m,n);

  double retval s2 - (pi_hat[id]*x_hat[ix]*z_hat[iz] - r1);

  if(gsl_isinf(retval) || gsl_isnan(retval))
    {
      printf("Aquisition FOC is INF or NAN!\n");
    }
  
  return retval;

}

int iterate_acquisition_policy(int id, double * maxdiff, int imaxdiff[3])
{
  *maxdiff = -HUGE_VAL;
    
  gsl_interp_accel * acc = gsl_interp_accel_alloc();
  
  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  for(int im=0; im<NM; im++)
	    {
	      gsl_interp_accel_reset(acc);
	      double gn_tmp = 0.0;
	      dp_params p = {id,ix,iz,im,acc};
	      double lower_bound=0.0;
	      double upper_bound = fmax(1.0-m_grid[im],m_grid[NM-1])-delta_root;

	      if(acquisition_foc(lower_bound,&p)>0.0)
		{
		  gn_tmp = 0.0;
		}
	      else if(entrant_foc(upper_bound,&p)<0.0)
		{
		  gn_tmp = upper_bound;
		}
	      else
		{
		  gsl_function f;
		  f.function = &entrant_foc;
		  f.params=&p;
		  if(find_root_1d(&f,lower_bound,upper_bound,&gn_tmp))
		    {
		      printf("\nError solving aquisition first-order condition! (id,ix,iz,im) = (%d,%d,%d,%id)\n",ix,iz,id,im);
		      gsl_interp_accel_free(acc);
		      return 1;
		    }
		}
	      
	      gs[id][ix][iz][im] = s(id,m_grid[im],gn_tmp);

	      double diff = fabs(gn_tmp-gn[id][ix][iz][im]) - gn[id][ix][iz][im]*policy_tol_rel;
	      gn[id][ix][iz][im] = gn_tmp;
	      
	      if(diff>*maxdiff)
		{
		  *maxdiff=diff;
		  imaxdiff[0]=ix;
		  imaxdiff[1]=iz;
		  imaxdiff[2]=im;
		}

	    }
	}
    }

  gsl_interp_accel_free(acc);
  return 0;

}

double retention_foc(double m3, void * data)
{
  dp_params * p = (dp_params *)data;
  int id = p->id;
  int ix = p->ix;
  int iz = p->iz;
  int im2 = p->im;
  gsl_interp_accel * acc = p->acc;

  int im3 = gsl_interp_accel_find(acc, m_grid, NM, m3);
  
  double EdV=0.0;
  for(int izp=0; izp<NZ; izp++)
    {
      if(z_trans_probs[iz][izp]>1.0e-11)
	{
	  double Vm = interp_with_ix(m_grid,dV[id][ix][izp],NM,m3,imp);

	  double Vm2=0.0;
	  if(rho_x<0.999)
	    {
	      Vm2 = interp_with_ix(m_grid,EdV_xp[id][izp],NM,m3,imp);
	    }
	  
	  EdV += z_trans_probs[iz][izp]*(rho_x*Vm + (1.0-rho_x)*Vm2);
	}
    }

  double m2 = m_grid[im];
  double r2 = dr_do(m2,m3);

  double retval = r2 - EdV;
  
  if(gsl_isinf(retval) || gsl_isnan(retval))
    {
      printf("Retention FOC is INF or NAN!\n");
    }

  return retval;
}

int iterate_retention_policy(int id, double * maxdiff, int imaxdiff[3])
{
  *maxdiff = -HUGE_VAL;
  
  gsl_interp_accel * acc = gsl_interp_accel_alloc();
  
  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{
	  for(im=0; im<NM; im++)
	    {
	      double m = m_grid[im];      
	      dp_params p = {id,ix,iz,im,acc};
	      double lower_bound = 0.0;
	      double upper_bound = m - delta_root;
	      double gm_tmp=0.0;
		  
	      if(retention_foc(lower_bound,&p)>0)
		{
		  gm_tmp = lower_bound;
		}
	      else if(incumbent_foc(upper_bound,&p)<0.0)
		{
		  gm_tmp = upper_bound;
		}
	      else
		{
		  gsl_function f;
		  f.function = &incumbent_foc;
		  f.params=&p;
		  if(find_root_1d(&f,lower_bound,upper_bound,&gm_tmp))
		    {
		      printf("\nError solving incumbent's first-order condition! (id,ix,iz,im) = (%d,%d,%d,%d)\n",ix,iz,id,im);
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
  double maxdiff1 = 999;
  int imaxdiff1[3] = {0}; 
  double maxdiff2 = 999;
  int imaxdiff2[3] = {0};
 
  int iter=0;
  do
    {
      iter++;
      calc_EdV(id);

      status = iterate_retention_policy(id,&maxdiff1,imaxdiff1);
      if(status)
	{
	  printf("\tError iterating retention policy function! id = %d\n",id);
	  break;
	}
      
      status = iterate_acquisition_policy_policy(id,&maxdiff2,imaxdiff2);
      if(status)
	{
	  printf("\tError iterating acquisition policy function! id = %d\n",id);
	  break;
	}

      // update envelope condition
      for(int ix=0; ix<NX; ix++)
	{
	  for(int iz=0; iz<NZ; iz++)
	    {
	      for(int im=0; im<NM; im++)
		{
		  dV[id][ix][iz][im] = pi_hat[id]*x_hat[ix]*z_hat[iz] -
		    ds_dm(m,gn[id][ix][iz][im]) - dr_dm(m+gn[id][ix][iz][im],gm[id][ix][iz][im]);
		}
	    }
	}

      if(verbose==4)
	{
	  printf("\t\tIter %d, diff1 = %0.2g, loc1 = (%d, %d, %d), gn[loc] = %0.4g, diff2 = %0.2g, loc2 = (%d, %d, %d), gm[loc] = %0.4g\n",
		 iter,
		 maxdiff1,imaxdiff1[0],imaxdiff1[1],imaxdiff1[2],gm[id][imaxdiff1[0]][imaxdiff1[1]][imaxdiff1[2]],
		 maxdiff2,imaxdiff2[0],imaxdiff2[1],imaxdiff2[2],gn[id][imaxdiff2[0]][imaxdiff2[1]][imaxdiff2[2]]);
	}
    }
  while(fmax(maxdiff1,maxdiff2)>policy_tol_abs && iter < policy_max_iter);

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

//////////////////////////////////////////////////////////////////////////////
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
  
  return 0;
}

int main(int argc, char * argv[])
{
 
  time_t start, stop;
  time(&start);

  // setup environment
  linebreak();
  linebreak();
  if(setup())
      return 1;
  
  linebreak();	  
  if(solve_policies(63))
    return 1;
 
}
