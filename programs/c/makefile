CC = gcc
CFLAGS = -Wall -pedantic -fopenmp -mcmodel=medium -ffast-math -DHAVE_INLINE=1 -D_MODEL_MAIN -g -O3

LDFLAGS = -fopenmp -lgsl -lgslcblas -lm -ldl -lgomp -lnlopt -g -O3  -L /usr/local/lib/ -Wl,-R/usr/local/lib/

EXDIR = bin
ODIR = src/obj

_OBJ = dyn_mkt_pen.o sunk_cost.o acr.o static_mkt_pen.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: src/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

dyn_mkt_pen: src/obj/dyn_mkt_pen.o
	$(CC) -o $(EXDIR)/$@ $^ $(CFLAGS) $(LDFLAGS)

sunk_cost: src/obj/sunk_cost.o
	$(CC) -o $(EXDIR)/$@ $^ $(CFLAGS) $(LDFLAGS)

acr: src/obj/acr.o
	$(CC) -o $(EXDIR)/$@ $^ $(CFLAGS) $(LDFLAGS)

static_mkt_pen: src/obj/static_mkt_pen.o
	$(CC) -o $(EXDIR)/$@ $^ $(CFLAGS) $(LDFLAGS)