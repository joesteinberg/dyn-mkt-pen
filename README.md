# Export Market Penetration Dynamics: Replication Package
**Joseph Steinberg, University of Toronto**

This repository contains the code required to reproduce the results in my paper "Export Market Penetration Dynamics" (currently R&R at the *Journal of International Economics*). It contains two folders at the top level: [data](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/data) and [programs](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs). The former contains the data required to replicate the empirical analysis (with some caveats, see the note below). The latter, which contains all of the programs that conduct the empirical and quantitative analyses, is split into two subfolders[programs/python](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/python) and [programs/c](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/c). The former contains all of the Python scripts required to replicate the empirical analysis in Section 2 of the paper, as well as the scripts that process the simulated model results to produce the tables and figures in Sections 4 and 5. The latter contains the code to simulate the calibrated model, which is written in C.

## 0. Notes and requirements ##

### 0.0 Note about the Brazilian microdata ###
The main data from Brazil used in this paper cannot legally be distributed, so the raw data files are not included in this repository. I have, however, included all of the key intermediate datasets required to reproduce the empirical results in Section 2. The processed firm-level data are in the file `bra_microdata_processed.pik` in the [programs/scripts/output/pik](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/python/output/pik) folder, and the files `bra_microdata_agg_by_d.pik` and `bra_microdata_agg_by_d2.pik` in the same folder contain data aggregated to the destination-year and destination levels (respectively). Starting with these files, one can reproduce all empirical results in Section 2 without the underlying raw data.

### 0.1. Note about other large files ###
There are also some other files that are omitted from the repository because they are too large to store here on Github. For example, the raw data from the World Bank Exporter Dynamics database used in Appendix C and the raw model simulation output. Any folder in this repository with omitted files contains a text file called `missing files.txt` listing them. I am happy to produce these files on request via SFTP or SCP. Please note that whenever possible, intermediate files are contained here so that this is in general not necessary.

### 0.2. Requirements ###
The scripts in the [programs/python](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/python) folder require Python version 3 and the NumPy, Pandas, Matplotlib, Statsmodels, Patsy, os, and sys libraries. The C code in the [programs/c](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/c) folder requires OpenMPI, GSL, and NLOpt. I ran all programs using Ubuntu Linux 19.10 on a 56-core AMD Threadripper workstation with 192GB of RAM; it may take a very long time (or may not be feasible at all) on a computer with fewer cores or less RAM. All commands listed below assume you are working in a Linux bash terminal. Please contact me if you need help running these programs on a different operating system.

## 1. Python scripts ##
Next to each file name, I have provided a brief description of what this script does and highlighted in bold the output files it produces. Intermediate datasets are stored in Python pickle format in the [programs/python/output/pik](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/python/output/pik) folder. These scripts should be run in order if one is starting from scratch, but otherwise one can generally rely on the intermediate outputs from previous scripts.

### 1.1 Scripts for downloading and processing data ###

`get_wbdata.py`: Pulls data from World Bank on GDP per capita and population for export destinations.

`gravdata.py`: Merges data from three sources: gravity variables from CEPII Gravity database; bilateral tariff data from TRAINS; and bilateral trade data from DOT. Computes trade barriers by running a gravity regression.

`bra_microdata_prep.py`: Processses the raw firm-level data from Brazil and merges on the World Bank and gravity variables. If you are starting from scratch, you must run this script before attempting to run the C program, as it produces a file with destination characteristics that the C program uses as an input.

### 1.2 Scripts for processing model output ###

`model_microdata_prep.py`: Processes the simulated firm-level data from the C program. Running this script without a command-line argument will process the data from the baseline model. Running it with the optional arguments `smp`, `sunkcost`, or `acr` processes the simulated data from the alternative models (static market penetration, sunk cost, and exogenous new exporter dynmamics models, respectively).  Running it with the optional arguments `abn1`, `abn2`, `abo1`, `abo2`, or `a0` processes the simulated data from the sensitivity analyses discussed in Appendix A.2 and A.3.

`model_mechanics_plots.py`: Produces **Figure 3** (fig3_policy_function_example.pdf). Output is stored in [programs/python/output/model_mech_figs](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/python/output/model_mech_figs).

`life_cycle_prep.py`: Conducts further processing on the files created by `bra_microdata_prep.py` and `model_microdata_prep.py`. Computes variables required to estimate equations (2)-(5) and stores in Stata format. Calls the following Stata do files to estimate these equations using the `reghdf` command: `life_cycle_data.do` (estimates for Brazilian data); `life_cycle_model.do` (estimates for baseline model); and `life_cycle_alt_models.do` (for alternative models and sensitivity analyses). All output (both intermediate datasets and estimation output) is stored in the folder [programs/python/output/stata](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/python/output/stata).

### 1.3 Scripts for creating tables and figures in paper ###

`sumstats_regs.py`: Conducts the empirical analyses in Section 2.1, some of the analysis comparing the models to data in Sections 4.3 and 4.4, and some of the sensitivity analyses in Appendices A.2 and A.3. Produces **Table 1** (table1_sumstats_regs.tex), **Table 3** (table3_model_results.tex), **Table 4** (table4_sumstats_regs_costs.tex), **Table A2** (tableA2_alpha_beta.tex), and **Table B1** (tableB1_sumstats_regs_cross_sec.tex). All output is stored in [programs/python/output/sumstats_regs](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/python/output/sumstats_regs).

`life_cycle_figs.py`: Loads intermediate output from previous script and creates **Table A.1** (tableA1_life_cycle_dyn_v_data_tests.tex), **Figure 1** (fig1_life_cycle_dyn_v_data.pdf), **Figure 2** (fig2_life_cycle_dyn_x_data.pdf), **Figure 4** (fig4_life_cycle_dyn_v_model.pdf), **Figure 5** (fig5_life_cycle_dyn_x_model.pdf), **Figure 6** (fig6_life_cycle_dyn_c_model.pdf), **Figure A.1** (figA1_life_cycle_dyn_v_alpha_beta.pdf), and **Figure A.2** (figA2_life_cycle_dyn_x_alpha_beta). All output is stored in [programs/python/output/life_cycle_figs](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/python/output/life_cycle_figs).

`transition_dynamics.py`: Loads aggregated transition dynamics files created by C program and creates **Figure 7** (fig7_tr_dyn_perm_tau_drop.pdf). Output is stored in [programs/python/output/transitions](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/python/output/transitions).

`by_nd.py': Conducts all additional analyses reported in Appendix B.2. Creates **Table B.2** (tableB2_drank_regs.tex), **Table B.3** (tableB3_exit_by_nd_drank.tex), **Figure B.1** (figB1_dist_by_nd_model_vs_data.pdf), **Figure B.2** (figB2_by_nd_drank_model_vs_data.pdf), and **Figure B.3** (figB3_cost_by_nd_drank.pdf). All output is stored in [programs/python/output/by_nd](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/python/output/by_nd).

### 1.4 Scripts for analyzing World Bank Exporter Dynamics Database data ###
Appendix C shows that the main empirical results also obtain in firm-level data from Mexico and Peru. This data is publicly available and can be legalluy distributed, unlike the Brazilian data usedd in the main text of the paper. The following scripts produce the results found in this appendix. These scripts are structured in much the same way as the ones listed above. Please note that all intermediate datasets and output files are stored in a single folder [programs/python/wbedd/output](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/python/app_wbedd/output).

`app_wbedd/wbedd_microdata_prep.py`: Processes World Bank Exporter Dynamics database microdata from Mexico and Peru.

`app_wbedd/sumstats_regs.py`: Creates **Table C.1** (tableC1_sumstats_regs_wbedd.tex).

`app_wbedd/life_cycle.py`: Computes variables required to estimate equations (2)-(5) and stores in Stata format. Calls the Stata do file `app_wbedd/life_cycle_data.do` to estimate these equations using the `reghdf` command. Creates **Figure C.1** (figC1_life_cycle_dyn_v_wbedd.pdf) and **Figure C.2**.(figC2_life_cycle_dyn_x_wbedd.pdf).

## C program ##
The program to solve the model is written in C. It uses OpenMP to parallelize the solution of the firm's problem and simulate microdatasets.

### Source code ###
All source code is contained in the folder [programs/c/src](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/c/src).

`dyn_mkt_pen.c`: Source code for baseline model and sensitivity analyses.

`static_mkt_pen.c`: Source code for static market penetration alternative model.

`sunk_cost.c`: Source code for sunk cost alternative model.

`acr.c`: Source code for exogenous new expoerter dynamics alternative model.

### Compiling and running the program ###
The source code is compiled by running `make` from the command line in the [programs/c](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/c) folder. To compile the baseline model, type `make dyn_mkt_pen` (or just `make`). To compile the alternative models, type `make static_mkt_pen`, `make sunk_cost`, or `make acr` respectively.  To run the models type `./bin/dyn_mkt_pen`, `./bin/static_mkt_pen`, `./bin/sunk_cost`, or `./bin/acr`. The baseline model program `./bin/dyn_mkt_pen` has several command line options that allow the user to perform sensitivity analyses as well as the results for the baseline calibration. 

### Output files ###
The output of the programs is contained in the folder [programs/c/output](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/c/output). The simulated microdata data are stored in files named `<x>_microdata.csv`, where `<x>` is the name of the model. The aggregated transition dynamics are stored in files named `tr_dyn_perm_tau_drop_<x>.csv` where again `<x>` denotes the name of the model.

