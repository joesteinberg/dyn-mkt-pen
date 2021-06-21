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
#include <gsl/gsl_linalg.h>
#include <omp.h>

// macros: discretization
#define NX 50 // fixed-effect grid size
#define NZ 101 // productivity shock grid size
#define ND 63 // number of destinations
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
const double policy_tol_abs = 1.0e-8;
const int policy_max_iter = 2000;
const double x_grid_ub_mult = 10.0;
const double x_grid_exp = 1.0;

// print verbose output y/n
const int verbose=1;

// initialize all elements of an array to the same numeric value
int cmpfunc(const void * a, const void * b)
{
  if ( *(double*)a <  *(double*)b ) return -1;
  else return 1;
}

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

/////////////////////////////////////////////////////////////////////////////
// 2. Declarations of parameters, grids, and inline functions
/////////////////////////////////////////////////////////////////////////////

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
double alpha = 0.0; // returns to population size in marketing to new customers
double alpha2 = 0.0; // returns to population size in marketing to new customers
double gama = 0.0; // diminishing returns to scale in marketing to new customers
double psi = 0.0; // marketing efficiency for new customers
double z_grid_mult_lb = 0.0;
double z_grid_mult_ub = 0.0;

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

double E_xhat_zhat[NX][NZ] = {{0.0}};

// destination-specific parameters
char name[ND][3] = {{""}}; // name
double L[ND] = {0.0}; // market size
double Y[ND] = {0.0}; // aggregate consumption index
double La[ND] = {0.0}; // = L^(alpha)
double Lam[ND] = {0.0}; // = L^(alpha-1)
double tau_hat[ND] = {0.0}; // = tau^(1-theta)
double ta[ND] = {1.0}; // = L^(alpha)
double tam[ND] = {1.0}; // = L^(alpha-1)
double pi_hat[ND] = {0.0}; // theta_hat*L*Y*tau_hat

// marketing cost for new customers
static inline double s(int id, double n)
{
  return La[id] * ta[id] * ( 1 - pow((1.0-n),1.0-gama) ) / psi / (1.0-gama);
}

// derivative of s wrt n
static inline double ds_dn(int id, double n)
{
  return La[id] * ta[id] / psi / pow((1.0-n),gama);
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

      for(int iz=0; iz<NZ; iz++)
	{
	  E_xhat_zhat[ix][iz] = 0.0;
	  for(int izp=0; izp<NZ; izp++)
	    {
	      double tmp = x_hat[ix]*z_hat[izp];
	      double tmp2 = 0.0;
	      for(int ixp=0; ixp<NX; ixp++)
		{
		  tmp2 += x_hat[ixp]*z_hat[izp]*x_ucond_probs[ixp];
		}
	      E_xhat_zhat[ix][iz] += z_trans_probs[iz][izp]*(rho_x*tmp + (1.0-rho_x)*tmp2);
	    }
	}
    }
}

// assigned parameters and initial guesses
int init_params()
{
  W = 1.0;
  Q = 0.86245704;
  theta = 5.0;
  theta_hat = (1.0/theta) * pow(theta/(theta-1.0),1.0-theta);

  delta0 = 34.65234;
  delta1 = 0.00309521;
  sig_x =  1.02;
  rho_x = 0.98194358;
  sig_z = 0.43933157;
  rho_z =  0.60418386;
  z_grid_mult_lb=3.56360171;
  z_grid_mult_ub=2.49261931;
  alpha = 0.50840453;
  alpha2=0.0;
  gama=6.43893845;
  psi = 0.09784021;

  //psi = 0.04;
  //alpha = 0.95;
  //gama = 15;
  //alpha = 1.1;
  //psi = 0.015;
  
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
	      La[id] = pow(L[id],alpha);
	      Lam[id] = pow(L[id],alpha-1.0);
	      ta[id] = pow(tau_hat[id],alpha2);
	      tam[id] = pow(tau_hat[id],alpha2-1.0);
	      strncpy(name[id],buffer,3);
	      //tau_hat[id] = pow(tau[id],1.0-theta);
	      pi_hat[id] = theta_hat * L[id] * Y[id] * tau_hat[id];
	    }
	}

      return 0;
    }
}


/////////////////////////////////////////////////////////////////////////////
// 3. Optimization problem
/////////////////////////////////////////////////////////////////////////////

double gm[ND][NX][NZ] = {{{0}}};
double gc[ND][NX][NZ] = {{{0}}};
double expart_rate[ND] = {0.0};
int policy_solved_flag[ND] = {0};

typedef struct
{
  int id;
  int ix;
  int iz;
}dp_params;

double entrant_foc(double mp, void * data)
{
  dp_params * p = (dp_params *)data;
  int id = p->id;
  int ix = p->ix;
  int iz = p->iz;

  double mc = ds_dn(id,mp);

  //double EV=0.0;
  
  return mc - pi_hat[id]*x_hat[ix]*z_hat[iz];
  //return mc - Q*delta[ix]*pi_hat[id]*E_xhat_zhat[ix][iz];
}

int solve_policies(int id)
{  
  for(int ix=0; ix<NX; ix++)
    {
      for(int iz=0; iz<NZ; iz++)
	{ 
	  dp_params p = {id,ix,iz};
	  double lower_bound=0.0;
	  double upper_bound=1.0-delta_root;

	  if(entrant_foc(0.0,&p)>0.0)
	    {
	      gm[id][ix][iz] = 0.0;
	      gc[id][ix][iz] = 0.0;
	    }
	  else if(entrant_foc(upper_bound,&p)<0.0)
	    {
	      gm[id][ix][iz] = upper_bound;
	      gc[id][ix][iz] = s(id,upper_bound);
	    }
	  else
	    {
	      gsl_function f;
	      f.function = &entrant_foc;
	      f.params=&p;
	      if(find_root_1d(&f,lower_bound,upper_bound,&(gm[id][ix][iz])))
		{
		  printf("\nError solving entrant's first-order condition! (id,ix,iz) = (%d,%d,%d)\n",ix,iz,id);
		  return 1;
		}
	      gc[id][ix][iz] = s(id,gm[id][ix][iz]);
	    }
	}
    }

  return 0;

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
      printf("Finished optimization problems in %0.0f seconds. %d failed to converge.\n",difftime(stop,start),cnt);
    }
  
  return 0;  
}

/////////////////////////////////////////////////////////////////////////////
// 4. Simulation
/////////////////////////////////////////////////////////////////////////////

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

  gsl_interp_accel * acc1 = gsl_interp_accel_alloc();
  gsl_interp_accel * acc2 = gsl_interp_accel_alloc();

  // then for each firm in the sample...
  for(int jf=0; jf<NF; jf++)
    {
      // find fixed-effect value based on random draw
      gsl_interp_accel_reset(acc1);
      int ix = gsl_interp_accel_find(acc1, x_ucond_cumprobs, NX, x_rand[jf][0]);

      if(ix<0 || ix>=NX)
	{
	  printf("Error!\n");
	}
      
      // find initial value of shock based on random draw and ergodic distribution
      gsl_interp_accel_reset(acc2);
      int iz = gsl_interp_accel_find(acc2, z_ucond_cumprobs, NZ, z_rand[id][jf][0]);
      iz_sim[id][jf][0] = iz;

      double m=0;
            
      for(int kt=0; kt<max_kt; kt++)
	{
	  // record current state variables
	  iz_sim[id][jf][kt]=iz;

	  if(id==0)
	    ix_sim[jf][kt]=ix;

	  double mp = gm[id][ix][iz];
	  
	  m_sim[id][jf][kt] = mp;

	  // record current exports (which depend on state variables only!)
	  if(mp<1.0e-8)
	    {
	      v_sim[id][jf][kt] = -99.9;
	    }
	  else
	    {
	      v_sim[id][jf][kt] = theta*theta_hat*L[id]*Y[id]*tau_hat[id]*x_hat[ix]*z_hat[iz]*mp;
	    }

	  // record exporting costs
	  if(mp<1.0e-8)
	    {
	      cost_sim[id][jf][kt] = -99.9;
	      cost2_sim[id][jf][kt] = -99.9;
	    }
	  else
	    {
	      double profit = v_sim[id][jf][kt]/theta;
	      double cost=0.0;
	      cost=gc[id][ix][iz];	      
	      cost_sim[id][jf][kt] = cost;
	      cost2_sim[id][jf][kt] = cost/profit;
	    }

	  // unless we are in the very last period of the simulation, update the state variables
	  if(kt<max_kt-1)
	    {
	      // if you die, set market penetration to zero and draw new shocks from ergodic distributions
	      if(surv_rand[jf][kt]>delta[ix])
		{
		  mp=0;
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

	      m=0;
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
      expart_rate_[kt-NT] = expart_rate_[kt-NT]/((double)(NF));
      avg_expart_rate += expart_rate_[kt-NT];
    }

  avg_expart_rate=avg_expart_rate/((double)(NT));
  expart_rate[id] = avg_expart_rate;
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
		     double * avg_top5_share)
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
      expart_rate_[kt-min_kt] = expart_rate_[kt-min_kt]/((double)(NF));
      *avg_expart_rate += expart_rate_[kt-min_kt];
    }

  *avg_expart_rate = *avg_expart_rate/(max_kt-min_kt);

  double exit_rate_[ND] = {0.0};
  double avg_nd_[ND] = {0.0};
  double top5_share_[ND] = {0.0};
  int ND2=0;
  for(int id=0; id<ND; id++)
    {
      if(policy_solved_flag[id]==0 && expart_rate[id]>1.0e-10)
	{
	  ND2++;
	  int NT2=0;
	  
	  for(int kt=min_kt; kt<max_kt; kt++)
	    {
	      int numex=0;
	      int exits=0;
	      int avg_nd2=0;

	      double vsorted[NF];
	      
	      for(int jf=0; jf<NF; jf++)
		{
		  if(v_sim[id][jf][kt]>1.0e-10)
		    {
		      vsorted[numex] = v_sim[id][jf][kt];
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
		}

	      if(numex>0)
		{
		  NT2++;
		  exit_rate_[id] += ((double)(exits))/((double)(numex));
		  avg_nd_[id] += ((double)(avg_nd2))/((double)(numex));

		  qsort(vsorted,numex,sizeof(double),cmpfunc);
		  double sum=0;
		  double sum2=0;
		  for(int iii=numex-1; iii>=0; iii--)
		    {
		      sum += vsorted[iii];
		      if( (double)(iii)/((double)(numex)) > 0.95 )
			sum2 += vsorted[iii];
		    }
		  top5_share_[id] += sum2/sum;
		}
	    }
	  if(NT2>0)
	    {
	      exit_rate_[id]= exit_rate_[id]/NT2;
	      avg_nd_[id]= avg_nd_[id]/NT2;
	      top5_share_[id]= top5_share_[id]/NT2;
	    }
	  else
	    {
	      exit_rate_[id]= 0.0;
	      avg_nd_[id]= 0.0;
	      top5_share_[id] = 0.0;
	      ND2--;
	      expart_rate[id]=0.0;
	    }
	}
    }

  *avg_exit_rate = 0.0;
  *avg_nd = 0.0;
  *avg_top5_share=0.0;
  for(int id=0; id<ND; id++)
    {
      if(policy_solved_flag[id]==0 && expart_rate[id]>1.0e-10)
	{
	  *avg_exit_rate += exit_rate_[id];
	  *avg_nd += avg_nd_[id];
	  *avg_top5_share += top5_share_[id];
	}
    }
  *avg_exit_rate = *avg_exit_rate/((double)(ND2));
  *avg_nd = *avg_nd/((double)(ND2));
  *avg_top5_share = *avg_top5_share/((double)(ND2));
	
  time(&stop);

  if(verbose)
    printf("Finished simulations in %0.0f seconds.\n\tOverall export part. rate = %0.8f\n\tavg. exit rate = %0.8f\n\tavg. num. dests = %0.8f\n\tavg. top 5 share = %0.8f\n",
	   difftime(stop,start),100*(*avg_expart_rate),100*(*avg_exit_rate),*avg_nd,*avg_top5_share);
    
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
  
  fprintf(file,"f,d,y,popt,gdppc,tau,v,m,cost,cost2,x,iz,nd,nd_group,entry,exit,incumbent,tenure,max_tenure,multilateral_exit\n");
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

double dist[ND][NX][NZ] = {{{0.0}}};
double tmp_gm[NT][ND][NX][NZ] = {{{{0.0}}}};
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
	  dist[id][ix][iz] = x_ucond_probs[ix] * z_ucond_probs[iz];
	  sum += dist[id][ix][iz];
	}
    }
  if(fabs(sum-1.0)>1.0e-8)
    {
      printf("\nInitial distribution does not sum to one! id = %d, sum = %0.4g\n",id,sum);
    }
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
	  if(gm[id][ix][iz] && dist[id][ix][iz]>1.0e-10)
	    {
	      m += gm[id][ix][iz] * x_hat[ix]*z_hat[iz]* dist[id][ix][iz];
	      //m += gm[id][ix][iz] * delta[ix]*E_xhat_zhat[ix][iz]* dist[id][ix][iz];
	      sumw += x_hat[ix]*z_hat[iz]* dist[id][ix][iz];
	      //sumw += delta[ix]*E_xhat_zhat[ix][iz]*dist[id][ix][iz];

	      expart_rate += delta[ix]*dist[id][ix][iz];
	      double v = delta[ix]*theta*theta_hat*L[id]*Y[id]*tau_hat[id]*E_xhat_zhat[ix][iz]*gm[id][ix][iz];
	      total_exports += dist[id][ix][iz] * v;
	    }
	}
    }

  m=m/sumw;
  tr_moments[0] = total_exports;
  tr_moments[1] = expart_rate;
  tr_moments[2] = m;
  
  return;
}

int tr_dyn_perm_tau_chg(int id, double chg)
{
  memcpy(tmp_gm[0][id],gm[id],sizeof(double)*NX*NZ);
  double tau_hat0 = tau_hat[id];  
  double tr_moments[3];

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
      calc_tr_dyn(id,tr_moments);
      tr_tau[id][t+1] = pow(tau_hat[id],1.0/(1.0-theta));
      tr_exports[id][t+1] = tr_moments[0];
      tr_expart[id][t+1] = tr_moments[1];
      tr_mktpen[id][t+1] = tr_moments[2];
      tr_te[id][t+1] = -log(tr_exports[id][t+1]/tr_exports[id][0])/log(1.0+chg);
    }

  memcpy(gm[id],tmp_gm[0][id],sizeof(double)*NX*NZ);
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

int tr_dyn_rer_shock(int id, double shock, double rho)
{
  double tau_hat0 = tau_hat[id];

  memcpy(tmp_gm[0][id],gm[id],sizeof(double)*NX*NZ);
  
  double tr_moments[3];

  // first solve policies backwards
  for(int t=NT-1; t>=1; t--)
    {
      double rer = exp(shock*pow(rho,t));
      tau_hat[id] = tau_hat0 * pow(rer,theta-1.0);
      pi_hat[id] = theta_hat * L[id] * Y[id] * tau_hat[id];
      solve_policies(id);
      memcpy(tmp_gm[t][id],gm[id],sizeof(double)*NX*NZ);
    }

  // now iterate forward

  // period 0: initial steady state
  memcpy(gm[id],tmp_gm[0][id],sizeof(double)*NX*NZ);
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
      memcpy(gm[id],tmp_gm[t][id],NX*NZ*sizeof(double));
      double rer = exp(shock*pow(rho,t));
      tau_hat[id] = tau_hat0 * pow(rer,theta-1.0);
      pi_hat[id] = theta_hat * L[id] * Y[id] * tau_hat[id];
      calc_tr_dyn(id,tr_moments);
      tr_tau[id][t+1] = pow(tau_hat[id],1.0/(1.0-theta));
      tr_exports[id][t+1] = tr_moments[0];
      tr_expart[id][t+1] = tr_moments[1];
      tr_mktpen[id][t+1] = tr_moments[2];
      tr_te[id][t+1] = -log(tr_exports[id][t+1]/tr_exports[id][0])/(-log(rer));
    }

  // go back to benchmark trade costs, policies, and dist
  memcpy(gm[id],tmp_gm[0][id],sizeof(double)*NX*NZ);
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
      fprintf(file,"d,popt,gdppc,t,tau,exports,expart_rate,mktpen_rate,trade_elasticity\n");
      for(int id=0; id<ND; id++)
	{
	  if(policy_solved_flag[id]==0)
	    {
	      for(int it=0; it<NT+1; it++)
		{
		  fprintf(file,"%.3s,%0.16f,%0.16f,%d,%0.16f,%0.16f,%0.16f,%0.16f,%0.16f\n",
			  name[id],L[id],Y[id],it,tr_tau[id][it],
			  tr_exports[id][it],tr_expart[id][it],
			  tr_mktpen[id][it],tr_te[id][it]);
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
  printf("Static market penetration model\n\n");
  
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
  random_draws();
  
  return 0;
}

int benchmark()
{
  printf("Solving and simulating model...\n");

  time_t start, stop;
  time(&start);

  if(solve_policies_all_dests())
    return 1;

  double expart_rate=0.0;
  double exit_rate=0.0;
  double avg_nd = 0.0;
  double avg_top5_share=0.0;
  simul_all_dests(&expart_rate,&exit_rate,&avg_nd,&avg_top5_share);

  create_panel_dataset("output/static_mkt_pen_microdata.csv");
  
  time(&stop);
  printf("\nBenchmark analysis complete! Runtime = %0.0f seconds.\n",
	 difftime(stop,start));
	  
  return 0;
}

int main(int argc, char * argv[])
{
  time_t start, stop;
  time(&start);

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
  for(int id=0; id<ND; id++)
    init_dist(id);
  
  // effects of permanent drop in trade costs
  linebreak();  
  if(tr_dyn_perm_tau_chg_all_dests(-0.1))
    return 1;

  if(write_tr_dyn_results("output/tr_dyn_perm_tau_drop_static_mkt_pen.csv"))
    return 1;  

  // effects of temporary good depreciation
  double shock = log(1.0+(theta-1.0)/10.0)/(theta-1.0);
  linebreak();
  if(tr_dyn_rer_shock_all_dests(shock,0.9))
    return 1;

  if(write_tr_dyn_results("output/tr_dyn_rer_dep_static_mkt_pen.csv"))
    return 1;

  linebreak();
  printf("\nCalling python scripts to process and analyze simulated microdata...\n");
  
  if(system("python3 -W ignore ../python/model_microdata_prep.py smp"))
    return 1;
  
  if(system("python3 -W ignore ../python/sumstats_regs.py smp"))
    return 1;

  if(system("python3 -W ignore ../python/life_cycle.py smp"))
    return 1;

  
  // finish program
  linebreak();  
  time(&stop);
  printf("\nProgram complete! Total runtime: %0.16f seconds.\n",difftime(stop,start));

  return 0;
}
#endif

