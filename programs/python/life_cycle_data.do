display "...Entering stata..."

set level 95
use output/stata/bra_microdata_processed, clear
encode d, generate(d2)
encode f, generate(f2)
gen logv = log(v)

/* Log sales */

* destination-year and firm-year fixed effects
reghdfe logv tenure#max_tenure censored_spell, absorb(d2#y f2#y)
regsave using output/stata/dreg_v_all, ci replace

* firm-year fixed effects with 3-way interaction
reghdfe logv tenure#max_tenure#grp censored_spell, absorb(f2#y)
regsave using output/stata/dreg_v_3way, ci replace

* save coefficient equality tests
file open testfile using output/stata/dreg_v_tests.txt, text write replace
file write testfile "tenure" _tab "max_tenure" _tab "H0_Fstat" _tab "H0_pval" _tab "H0_pval_1sided" _tab "diff" _tab "diff_se" _tab "diff_t" _tab "diff_p" _tab "diff_lb" _tab "diff_ub" _n
forval i = 0/5 {
forval j = 0/`i' {
local sgn = 0
if `i' == 0  & `j'== 0 {
test 0b.tenure#0b.max_tenure#0b.grp = 0b.tenure#0b.max_tenure#1.grp
local sgn = sign(_b[0b.tenure#0b.max_tenure#0b.grp] - _b[0b.tenure#0b.max_tenure#1.grp])
}
else if `i'==0 & `j'>0 {
test `j'.tenure#0b.max_tenure#0b.grp = `j'.tenure#0b.max_tenure#1.grp
local sgn = sign(_b[`j'.tenure#0b.max_tenure#0b.grp] - _b[`j'.tenure#0b.max_tenure#1.grp])
}
else if `i'>0 & `j'==0 {
test 0b.tenure#`i'.max_tenure#0b.grp = 0b.tenure#`i'.max_tenure#1.grp
local sgn = sign(_b[0b.tenure#`i'.max_tenure#0b.grp] - _b[0b.tenure#`i'.max_tenure#1.grp])
}
else {
test `j'.tenure#`i'.max_tenure#0b.grp = `j'.tenure#`i'.max_tenure#1.grp
local sgn = sign(_b[`j'.tenure#`i'.max_tenure#0b.grp] - _b[`j'.tenure#`i'.max_tenure#1.grp])
}
file write testfile %1s ("`j'") _tab %1s ("`i'") _tab %9.3f (r(F)) _tab %9.8g (r(p)) _tab %9.8g (normal(`sgn'*sqrt(r(F))))

if `i' == 0  & `j'== 0 {
lincom 0b.tenure#0b.max_tenure#1.grp - 0b.tenure#0b.max_tenure#0b.grp
}
else if `i'==0 & `j'>0 {
lincom `j'.tenure#0b.max_tenure#1.grp - `j'.tenure#0b.max_tenure#0b.grp
}
else if `i'>0 & `j'==0 {
lincom 0b.tenure#`i'.max_tenure#1.grp - 0b.tenure#`i'.max_tenure#0b.grp
}
else {
lincom `j'.tenure#`i'.max_tenure#1.grp - `j'.tenure#`i'.max_tenure#0b.grp
}
file write testfile _tab %9.8g (r(estimate)) _tab %9.8g (r(se)) _tab %9.8g (r(t)) _tab %9.8g (r(p)) _tab %9.8g (r(lb)) _tab %9.8g (r(ub))  _n

}
}

/* https://www.stata.com/support/faqs/statistics/one-sided-tests-for-coefficients/ */
/* https://www.statalist.org/forums/forum/general-stata-discussion/general/1339155-hypotesis-testing-one-coefficient-larger-then-the-other */


file close testfile

		 

/* Conditional exit */

drop if censored==1

reghdfe xit i.tenure, absorb(d2#y f2#y)
regsave using output/stata/dreg_x_all, ci replace

* firm-year fixed effects with 2-way interaction
reghdfe xit tenure#grp, absorb(f2#y)
regsave using output/stata/dreg_x_2way, ci replace

clear

display "...Finished! Exiting stata..."
