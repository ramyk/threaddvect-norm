##
# threaddvect-norm
#
# @src/makefile
# @version 0.1

IDIR =../include
CC = gcc
CFLAGS = -mavx -I$(IDIR)

# .o output files directory
ODIR = obj

# external libraries linking
LIBS = -lm -lpthread

_DEPS = norm.h
DEPS = $(patsubst %, $(IDIR)/%, $(_DEPS))

_OBJ = main.o norm.o
OBJ = $(patsubst %, $(ODIR)/%, $(_OBJ))

$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

parallel_norm.out: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS) $(MACHINE)

# this .PHONY rule keeps make from doing
# something with a file named clean
.PHONY: clean

# pre-compile cleaning rule
clean:
	/bin/rm -f $(ODIR)/*.o *~ *.out $(IDIR)/*~

# end
