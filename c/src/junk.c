    /*const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
      gsl_multifit_nlinear_workspace *w;
      gsl_multifit_nlinear_fdf fdf;
      gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();
      fdf_params.h_df = 1.0e-4;
      //fdf_params.solver = gsl_multifit_nlinear_solver_;
      const size_t n = NY;
      const size_t p = NP;
      double weights[NY]={1.0};
      
      FILE * file = fopen("../python/output/calibration_data.txt","r");
      if(!file)
	{
	  printf("Failed to open file with calibration data!\n");
	  return 1;
	}
      else
	{
	  double data_moments[NY];
	  double model_moments[NY];
	  
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
	      /*
		In [62]: mu
		Out[62]: 0.3402940345625788
		
		In [63]: se
		Out[63]: 0.008915307195321662
		
		exit
		mean        se
		nd_group
		1         0.566251  0.006238
		2         0.375161  0.006970
		3         0.245397  0.005813
		4         0.164864  0.004722
		6         0.087656  0.003073
		10        0.023675  0.001910*/

	      weights[NY-8] = 0.1;
	      weights[NY-7] = 0.008915307195321662;
	      weights[NY-6] = 0.006238;
	      weights[NY-5] = 0.006970;
	      weights[NY-4] = 0.005813;
	      weights[NY-3] = 0.004722;
	      weights[NY-2] = 0.003073;
	      weights[NY-1] = 0.001910;
	    }
	}

      for(int i=0; i<NY; i++)
	{
	  //weights[i] = 1.0/pow(weights[i],2.0);
	  weights[i] = 1.0;
	}
      
      gsl_vector *f;
      gsl_matrix *J;
      gsl_matrix *covar = gsl_matrix_alloc (p, p);
      gsl_vector_view x = gsl_vector_view_array (params, p);
      gsl_vector_view wts = gsl_vector_view_array(weights, n);
      double chisq, chisq0;
      int status, info;

      const double xtol = 1e-8;
      const double gtol = 1e-8;
      const double ftol = 0.0;

      results_file = fopen("output/results.csv","a");

      /* define the function to be minimized */
      fdf.f = work_gsl_wrapper;
      fdf.df = NULL;   /* set to NULL for finite-difference Jacobian */
      fdf.fvv = NULL;     /* not using geodesic acceleration */
      fdf.n = n;
      fdf.p = p;
      fdf.params = NULL;

      /* allocate workspace with default parameters */
      w = gsl_multifit_nlinear_alloc (T, &fdf_params, n, p);

      /* initialize solver with starting point and weights */
      gsl_multifit_nlinear_winit (&x.vector, &wts.vector, &fdf, w);
      
      /* compute initial cost function */
      f = gsl_multifit_nlinear_residual(w);
      gsl_blas_ddot(f, f, &chisq0);
      
      /* solve the system with a maximum of 100 iterations */
      status = gsl_multifit_nlinear_driver(100, xtol, gtol, ftol,
					   callback, NULL, &info, w);

      /* compute covariance of best fit parameters */
      J = gsl_multifit_nlinear_jac(w);
      gsl_multifit_nlinear_covar (J, 0.0, covar);
      
      /* compute final cost */
      gsl_blas_ddot(f, f, &chisq);

      linebreak();
      linebreak();
      
      fprintf(stderr, "summary from method '%s/%s'\n",
	      gsl_multifit_nlinear_name(w),
	      gsl_multifit_nlinear_trs_name(w));
      fprintf(stderr, "number of iterations: %zu\n",
	      gsl_multifit_nlinear_niter(w));
      fprintf(stderr, "function evaluations: %zu\n", fdf.nevalf);
      fprintf(stderr, "Jacobian evaluations: %zu\n", fdf.nevaldf);
      fprintf(stderr, "reason for stopping: %s\n",
	      (info == 1) ? "small step size" : "small gradient");
      fprintf(stderr, "initial |f(x)| = %f\n", sqrt(chisq0));
      fprintf(stderr, "final   |f(x)| = %f\n", sqrt(chisq));
      
      double dof = n - p;
      double c = GSL_MAX_DBL(1, sqrt(chisq / dof));
      fprintf(stderr, "chisq/dof = %g\n", chisq / dof);

#define FIT(i) gsl_vector_get(w->x, i)
#define ERR(i) sqrt(gsl_matrix_get(covar,i,i))

      sig_x = FIT(0);
      rho_x = FIT(1);
      sig_z = FIT(2);
      rho_z = FIT(3);
      psi_n = FIT(4);
      alpha_n = FIT(5);
      gamma_n = FIT(6);
      psi_o = FIT(7);
      alpha_o = FIT(8);
      gamma_o = FIT(9);
      delta0 = FIT(10);
      delta1 = FIT(11);
      beta_n = FIT(12);
      beta_o = FIT(13);

      printf("\nCalibrated parameter vector:\n");
      printf("\tsig_x =     %0.8f +/- %.8f\n",sig_x,c*ERR(1));
      printf("\trho_x =     %0.8f +/- %.8f\n",rho_x,c*ERR(2));
      printf("\tsig_z =     %0.8f +/- %.8f\n",sig_z,c*ERR(3));
      printf("\trho_z =     %0.8f +/- %.8f",rho_z,c*ERR(4));
      printf("\tpsi_n =     %0.8f +/- %.8f",psi_n,c*ERR(5));
      printf("\talpha_n =   %0.8f +/- %.8f",alpha_n,c*ERR(6));
      printf("\tbeta_n =    %0.8f +/- %.8f",beta_n,c*ERR(7));
      printf("\tgamma_n =   %0.8f +/- %.8f",gamma_n,c*ERR(8));
      printf("\tpsi_o =     %0.8f +/- %.8f",psi_o,c*ERR(9));
      printf("\talpha_o =   %0.8f +/- %.8f",alpha_o,c*ERR(10));
      printf("\tbeta_o =    %0.8f +/- %.8f",beta_o,c*ERR(11));
      printf("\tgamma_o =   %0.8f +/- %.8f",gamma_o,c*ERR(12));
      printf("\tdelta0 =    %0.8f +/- %.8f",delta0,c*ERR(13));
      printf("\tdelta1 =    %0.8f +/- %.8f\n",delta1,c*ERR(14));
      
      printf("status = %s\n", gsl_strerror (status));
      linebreak();
      linebreak();

      gsl_multifit_nlinear_free (w);
      gsl_matrix_free (covar);
      fclose(results_file);*/



      ork_nlopt_wrapper(unsigned n, const double *x, double *grad, void *my_func_data)
{
  if(x->size != NP || f->size != NY)
    {
      return 1;
    }
  else
    {
      double xv[NP];
      double fv[NY];
      for(int i=0; i<NP; i++)
	{
	  xv[i] = gsl_vector_get(x,i);
	}
      if(work(xv,fv))
	{
	  return 1;
	}
      
      for(int i=0; i<NY; i++)
	{
	  gsl_vector_set(f,i,fv[i]);
	  //printf("\tF(%d) = %0.8f\n",i,gsl_vector_get(f,i));
	}
      return 0;
    }
    }*/


    void callback(const size_t iter, void *params,
         const gsl_multifit_nlinear_workspace *w)
{
  gsl_vector *f = gsl_multifit_nlinear_residual(w);
  gsl_vector *x = gsl_multifit_nlinear_position(w);
  gsl_matrix * J = gsl_multifit_nlinear_jac(w);
  double rcond;

  /* compute reciprocal condition number of J(x) */
  gsl_multifit_nlinear_rcond(&rcond, w);

  linebreak();
  linebreak();
  printf("Iter %2zu: cond(J) = %8.4f, |f(x)| = %.4f\n",
	 iter,
	 1.0 / rcond,
	 gsl_blas_dnrm2(f));

  /*printf("\nJacobian:\n");
  for(int i=0; i<NY; i++)
    {
      for(int j=0; j<NP; j++)
	{
	  printf(" %0.5f",gsl_matrix_get(J,i,j));
	}
      printf("\n");
      }*/

  
  linebreak();
  linebreak();

}






	      printf("Scatter plots:\n");
	      printf("\tTop 5 share (avg):             %0.4f %0.4f %0.4f\n",
		     data_moments[0],model_moments[0],residuals[0]);
	      printf("\tTop 5 share (slope):           %0.4f %0.4f %0.4f\n",
		     data_moments[1],model_moments[1],residuals[1]);
	      printf("\tAvg num dest (avg):            %0.4f %0.4f %0.4f\n",
		     data_moments[2],model_moments[2],residuals[2]);
	      printf("\tAvg num dest (slope):         %0.4f %0.4f %0.4f\n",
		     data_moments[3],model_moments[3],residuals[3]);
	      printf("\tExit rate (avg):               %0.4f %0.4f %0.4f\n",
		     data_moments[4],model_moments[4],residuals[4]);
	      printf("\tExit rate (slope):            %0.4f %0.4f %0.4f\n",
		     data_moments[5],model_moments[5],residuals[5]);
	      printf("\tEntrant rel size (avg):        %0.4f %0.4f %0.4f\n",
		     data_moments[6],model_moments[6],residuals[6]);
	      printf("\tEntrant rel size (slope):     %0.4f %0.4f %0.4f\n",
		     data_moments[7],model_moments[7],residuals[7]);
	      printf("\tEntrant rel exit rate (avg):   %0.4f %0.4f %0.4f\n",
		     data_moments[8],model_moments[8],residuals[8]);
	      printf("\tEntrant rel exit rate (slope): %0.4f %0.4f %0.4f\n",
		     data_moments[9],model_moments[9],residuals[9]);

	      printf("Effects of tenure on survival (hard)\n");
	      printf("\t1:                            %0.4f %0.4f %0.4f\n",
		     data_moments[10],model_moments[10],residuals[10]);
	      printf("\t2:                            %0.4f %0.4f %0.4f\n",
		     data_moments[11],model_moments[11],residuals[11]);
	      printf("\t3:                            %0.4f %0.4f %0.4f\n",
		     data_moments[12],model_moments[12],residuals[12]);
	      printf("\t4:                            %0.4f %0.4f %0.4f\n",
		     data_moments[13],model_moments[13],residuals[13]);
	      printf("\t5:                            %0.4f %0.4f %0.4f\n",
		     data_moments[14],model_moments[14],residuals[14]);

	      printf("\nEffects of tenure on survival (easy)\n");
	      printf("\t1:                            %0.4f %0.4f %0.4f\n",
		     data_moments[15],model_moments[15],residuals[15]);
	      printf("\t2:                            %0.4f %0.4f %0.4f\n",
		     data_moments[16],model_moments[16],residuals[16]);
	      printf("\t3:                            %0.4f %0.4f %0.4f\n",
		     data_moments[17],model_moments[17],residuals[17]);
	      printf("\t4:                            %0.4f %0.4f %0.4f\n",
		     data_moments[18],model_moments[18],residuals[18]);
	      printf("\t5:                            %0.4f %0.4f %0.4f\n",
		     data_moments[19],model_moments[19],residuals[19]);

	      printf("\nEffects of tenure:duration on sales (hard)\n");
	      int i=20;
	      for(int duration=1; duration<=5; duration++)
		{
		 for(int tenure=0; tenure<=duration; tenure++)
		   {
		    printf("\t%d:%d                            %0.4f %0.4f %0.4f\n",
			   tenure,duration,data_moments[i],model_moments[i],residuals[i]);
		    i++;
		   }
		}

	      printf("\nEffects of tenure:duration on sales (easy)\n");
	      for(int duration=1; duration<=5; duration++)
		{
		 for(int tenure=0; tenure<=duration; tenure++)
		   {
		    printf("\t%d:%d                            %0.4f %0.4f %0.4f\n",
			   tenure,duration,data_moments[i],model_moments[i],residuals[i]);
		    i++;
		   }
		}

	      
	      printf("\nMultilateral ex. part. rate:           %0.4f %0.4f %0.4f\n",
		     data_moments[i],model_moments[i],residuals[i]);
	      printf("Multilateral exit rate:                %0.4f %0.4f %0.4f\n",
		     data_moments[i+1],model_moments[i+1],residuals[i+1]);
	      printf("Multilateral exit rate (1D):           %0.4f %0.4f %0.4f\n",
		     data_moments[i+2],model_moments[i+2],residuals[i+2]);
	      printf("Multilateral exit rate (2D):           %0.4f %0.4f %0.4f\n",
		     data_moments[i+3],model_moments[i+3],residuals[i+3]);
	      printf("Multilateral exit rate (3D):           %0.4f %0.4f %0.4f\n",
		     data_moments[i+4],model_moments[i+4],residuals[i+4]);
	      printf("Multilateral exit rate (4D):           %0.4f %0.4f %0.4f\n",
		     data_moments[i+5],model_moments[i+5],residuals[i+5]);
	      printf("Multilateral exit rate (6D):           %0.4f %0.4f %0.4f\n",
		     data_moments[i+6],model_moments[i+6],residuals[i+6]);
	      printf("Multilateral exit rate (10D):          %0.4f %0.4f %0.4f\n",
		     data_moments[i+7],model_moments[i+7],residuals[i+7]);
