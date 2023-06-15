display "...Entering stata..."

use output/stata/model_microdata_processed, clear
encode d, generate(d2)
encode f, generate(f2)
gen logv = log(v)
gen logc = log(cost)

/* Log sales */
reghdfe logv tenure#max_tenure, absorb(d2#y f2#y)
regsave using output/stata/sreg_v_all, ci replace

reghdfe logv tenure#max_tenure#grp, absorb(f2#y)
regsave using output/stata/sreg_v_3way, ci replace

/* Conditional exit */
reghdfe xit i.tenure, absorb(d2#y f2#y)
regsave using output/stata/sreg_x_all, ci replace
	 
reghdfe xit tenure#grp, absorb(f2#y)
regsave using output/stata/sreg_x_2way, ci replace

	 
/* Log export costs */
reghdfe logc tenure#max_tenure, absorb(d2#y f2#y)
regsave using output/stata/sreg_c_all, ci replace

reghdfe logc tenure#max_tenure#grp, absorb(f2#y)
regsave using output/stata/sreg_c_3way, ci replace

/* Costs/profits */
reghdfe cost2 tenure#max_tenure, absorb(d2#y f2#y)
regsave using output/stata/sreg_c2_all, ci replace

reghdfe cost2 tenure#max_tenure#grp, absorb(f2#y)
regsave using output/stata/sreg_c2_3way, ci replace


clear

display "...Finished! Exiting stata..."
