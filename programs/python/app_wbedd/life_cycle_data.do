display "...Entering stata..."

set level 90
use output/mex_microdata_processed, clear
encode d, generate(d2)
encode f, generate(f2)
gen logv = log(v)

/* Log sales */

* destination-year and firm-year fixed effects
reghdfe logv tenure#max_tenure censored_spell, absorb(d2#y f2#y)
regsave using output/mreg_v_all, ci replace

* firm-year fixed effects with 3-way interaction
reghdfe logv tenure#max_tenure#grp censored_spell, absorb(f2#y)
regsave using output/mreg_v_3way, ci replace

/* Conditional exit */

drop if censored==1

reghdfe xit i.tenure, absorb(d2#y f2#y)
regsave using output/mreg_x_all, ci replace

* firm-year fixed effects with 2-way interaction
reghdfe xit tenure#grp, absorb(f2#y)
regsave using output/mreg_x_2way, ci replace

clear


use output/per_microdata_processed, clear
encode d, generate(d2)
encode f, generate(f2)
gen logv = log(v)

/* Log sales */

* destination-year and firm-year fixed effects
reghdfe logv tenure#max_tenure censored_spell, absorb(d2#y f2#y)
regsave using output/preg_v_all, ci replace

* firm-year fixed effects with 3-way interaction
reghdfe logv tenure#max_tenure#grp censored_spell, absorb(f2#y)
regsave using output/preg_v_3way, ci replace

/* Conditional exit */

drop if censored==1

reghdfe xit i.tenure, absorb(d2#y f2#y)
regsave using output/preg_x_all, ci replace

* firm-year fixed effects with 2-way interaction
reghdfe xit tenure#grp, absorb(f2#y)
regsave using output/preg_x_2way, ci replace

clear



display "...Finished! Exiting stata..."
