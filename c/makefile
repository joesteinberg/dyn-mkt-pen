.PHONY: clean

clean:
	rm -f src/obj/*.o
	rm -f bin/*

model:
	make dyn_mkt_pen -f makefile.model

calibrate:
	make calibrate_dyn_mkt_pen -f makefile.calibrate
