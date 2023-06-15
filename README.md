# Export Market Penetration Dynamics: Replication Package
Joseph Steinberg, University of Toronto

This repository contains the code required to reproduce the results in my paper "Export Market Penetration Dynamics" (currently R&R at the *Journal of International Economics*). It contains two folders at the top level: (data.md) and (programs.md). The former contains the data required to replicate the empirical analysis (with some caveats, see the note below). The latter, which contains all of the programs that conduct the empirical and quantitative analyses, is split into two subfolders `programs/python` and `programs/c`. The former contains all of the Python scripts required to replicate the empirical analysis in Section 2 of the paper, as well as the scripts that process the simulated model results to produce the tables and figures in Sections 4 and 5. The latter contains the code to simulate the calibrated model, which is written in C.

## Note about the Brazilian microdata ##
The main data used in this paper cannot legally be distributed, so the raw data files are not included in this repository. I have, however, included all of the key intermediate datasets required to reproduce the empirical results in Section 2. The processed firm-level data are in the file `bra_microdata_processed.pik` in the `programs/scripts/output/pik' folder, and the files `bra_microdata_agg_by_d.pik' and `bra_microdata_agg_by_d2.pik' contain data aggregated to the destination-year level.

## Note about other large files ##
There are also some other files that are omitted from the repository because they are too large to store here on Github. For example, the raw data from the World Bank Exporter Dynamics database used in Appendix C and the raw model simulation output. Any folder in this repository with omitted files contains a text file called `missing files.txt` listing them. I am happy to produce these files on requests to interested researchers via SFTP or SCP. Please note that whenever possible, intermediate files are contained here so that this is not necessary.

## Python scripts ##
The scripts in the `programs/python` folder require
