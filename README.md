# Export Market Penetration Dynamics: Replication Package
Joseph Steinberg, University of Toronto

This repository contains the code required to reproduce the results in my paper "Export Market Penetration Dynamics" (currently R&R at the *Journal of International Economics*). It contains two folders at the top level: [data](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/data) and [programs](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs). The former contains the data required to replicate the empirical analysis (with some caveats, see the note below). The latter, which contains all of the programs that conduct the empirical and quantitative analyses, is split into two subfolders[programs/python](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/python) and [programs/c](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/c). The former contains all of the Python scripts required to replicate the empirical analysis in Section 2 of the paper, as well as the scripts that process the simulated model results to produce the tables and figures in Sections 4 and 5. The latter contains the code to simulate the calibrated model, which is written in C.

## Note about the Brazilian microdata ##
The main data from Brazil used in this paper cannot legally be distributed, so the raw data files are not included in this repository. I have, however, included all of the key intermediate datasets required to reproduce the empirical results in Section 2. The processed firm-level data are in the file `bra_microdata_processed.pik` in the [programs/scripts/output/pik](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/python/output/pik) folder, and the files `bra_microdata_agg_by_d.pik' and `bra_microdata_agg_by_d2.pik' in the same folder contain data aggregated to the destination-year level. Starting with these files, one can reproduce all results in the paper without the underlying raw data.

## Note about other large files ##
There are also some other files that are omitted from the repository because they are too large to store here on Github. For example, the raw data from the World Bank Exporter Dynamics database used in Appendix C and the raw model simulation output. Any folder in this repository with omitted files contains a text file called `missing files.txt` listing them. I am happy to produce these files on request via SFTP or SCP. Please note that whenever possible, intermediate files are contained here so that this is in general not necessary.

## Requirements ##
The scripts in the [programs/python](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/python) folder require Python version 3 and the NumPy, Pandas, Matplotlib, and Statsmodels libraries. The C code in the [programs/c](https://github.com/joesteinberg/dyn-mkt-pen/tree/main/programs/c) folder requires OpenMPI, GSL, and NLOpt. I ran all programs using Ubuntu Linux 19.10 on a 56-core AMD Threadripper workstation with 192GB of RAM. All commands listed below assume you are working in a Linux bash terminal. Please contact me if you need help running these programs on a different operating system.

## Python scripts ##
Next to each file name, I have provided a brief description of what this script does and highlighted in bold the output files it produces. These scripts should be run in order if one is starting from scratch, but otherwise one can generally rely on the intermediate outputs from previous scripts.

`get_wbdata.py`: Pulls data from World Bank on GDP per capita and population for export destinations.
`gravdata.py`: Merges data from three sources: gravity variables from CEPII Gravity database; bilateral tariff data from TRAINS; and bilateral trade data from DOT. Computes trade barriers by running a gravity regression.
`bra_microdata_prep.py`: Processses the raw firm-level data from Brazil and merges on the World Bank and gravity variables. If you are starting from scratch, you must run this script before attempting to run the C program, as it produces a file with destination characteristics that the C program uses as an input.
`model_microdata.
