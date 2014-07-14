
# just a basic simple makefile.  If you want to use something other than
# gfortran, just change it

all: torch

torch: torch.f weaktab.f
	gfortran -O3 -o torch weaktab.f torch.f
