display "...Entering stata..."

foreach x in smp sunkcost acr {

use output/stata/`x'_microdata_processed, clear
encode d, generate(d2)
encode f, generate(f2)
gen logv = log(v)

/* Log sales */
reghdfe logv tenure#max_tenure, absorb(d2#y f2#y)
regsave using output/stata/sreg_v_all_`x', ci replace

reghdfe logv tenure#max_tenure#grp, absorb(f2#y)
regsave using output/stata/sreg_v_3way_`x', ci replace

/* Conditional exit */
reghdfe xit i.tenure, absorb(d2#y f2#y)
regsave using output/stata/sreg_x_all_`x', ci replace

reghdfe xit tenure#grp, absorb(f2#y)
regsave using output/stata/sreg_x_2way_`x', ci replace

clear

}


foreach x in abn1 abn2 abo1 abo2 a0 {

use output/stata/`x'_microdata_processed, clear
encode d, generate(d2)
encode f, generate(f2)
gen logv = log(v)

/* Log sales */
reghdfe logv tenure#max_tenure, absorb(d2#y f2#y)
regsave using output/stata/sreg_v_all_`x', ci replace

reghdfe logv tenure#max_tenure#grp, absorb(f2#y)
regsave using output/stata/sreg_v_3way_`x', ci replace

/* Conditional exit */

reghdfe xit i.tenure, absorb(d2#y f2#y)
regsave using output/stata/sreg_x_all_`x', ci replace

reghdfe xit tenure#grp, absorb(f2#y)
regsave using output/stata/sreg_x_2way_`x', ci replace

clear

}


display "...Finished! Exiting stata..."
