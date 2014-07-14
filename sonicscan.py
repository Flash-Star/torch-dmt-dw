import os
import sys
import subprocess
import math
import numpy as np

# Begin some global definitions
#xini = {'c12': 0.5, 'o16': 0.48, 'ne20': 0.0, 'ne22': 0.02,
#	'na23': 0.0, 'mg24': 0.0}

xini = {'c12': 0.15, 'o16': 0.45, 'ne20': 0.35, 'ne22': 0.01,
        'na23': 0.05, 'mg24': 0.05}

xfin = {'c12': 0.05, 'o16': 0.45, 'ne20': 0.45, 'ne22': 0.01,
        'na23': 0.05, 'mg24': 0.05}

xall = {'n': 0.0, 'h1': 0.0, 'he4': 0.0, 'c12': 0.0, 'c13': 0.0,
	'n14': 0.0, 'o16': 0.0, 'ne20': 0.0, 'ne22': 0.0, 
	'na23': 0.0, 'mg24': 0.0, 'si28': 0.0, 'fe52': 0.0,	
	'fe54': 0.0, 'fe56': 0.0, 'ni56': 0.0} 

vdetopt = 1057232460.76
maxvit = 1000
roughvit = 6
vdettol = 0.0001e9
fdvdet = 0.2

# Refinement settings: note that the top-level is currentrefine=0
maxrefine = 4
currentrefine = 0
refpoints = 10 # Factor by which to shrink the w spacing

torchoutlabel = 'w'
wlo = 0.0
whi = 1.0
# Number of top-level points in w to use
wnum = 100

os.chdir('./torch-dmt-dw')
mainlog = open('sonicscan.log','w')
fsr = open('scanrange_log.txt','w')
# End global definitions

def main():
	global fsr
	global xall
	global wnum
	global wlo
	global whi

        xk = xall.keys()

        # Write log header
        fsr.write('w vdet ')
        for k in xk:
                fsr.write(k + ' ')
        fsr.write('\n')

	#scanrange(wlo,whi,wnum)
	bruteforce(1.0e9,1.2e9,wnum,1.0)
	closelogs()

def criticalhalt():
	writetolog('Critical halt!')
	closelogs()
	sys.exit()

def closelogs():
	global fsr
	global mainlog
	fsr.close()
	mainlog.close()

def bruteforce(vdetlo,vdethi,numpts,w):
        global xall

        computex(w)

        foundgoodnan = False
        foundgoodovr = False
        nanvdet = 0.0
        ovrvdet = 0.0

	bf = open('bruteforce.log','w')
	bf.write('vdetk     isnan\n')

        for k in range(0,numpts+1):
                vdetk = math.pow(10.0,(math.log10(vdetlo) + float(k)*
                        (math.log10(vdethi)-math.log10(vdetlo))
                        /float(numpts)))
                errstate = testvdetsonic(vdetk)
                if not hascriticalerror(errstate):
                        if hasnan(errstate):
                                foundgoodnan = True
                                writetolog('Found nan state. vdetk = ' + str(vdetk))
				bf.write(str(vdetk) + '     1\n')
                                nanvdet = vdetk
                        else:
                                foundgoodovr = True
                                writetolog('Found ovr state. vdetk = ' + str(vdetk))
				bf.write(str(vdetk) + '     0\n')
                                ovrvdet = vdetk
#                if (foundgoodnan and foundgoodovr and (nanvdet < ovrvdet)):
#                        vdet = subdividevdet(ovrvdet,nanvdet)
#                        foundvdet(xall['c12'],vdet)
#                        return
#        writetolog('Could not find vdet.')
	bf.close()
        return

def scanrange(wmin,wmax,n):
	# First check the current refinement level...
	global maxrefine
	global currentrefine
	if (currentrefine > maxrefine):
		writetolog('Error: maxrefine exceeded!')
		criticalhalt()	

	# Iterate through the given range of w, finding vdet
	# Return the vdet corresponding to wmax.
	# w domain is n total points in: [wmin,wmax]	

	global fsr
	global xall
	global vdetopt
	global fdvdet
	global torchoutlabel

	wmin = float(wmin)
	wmax = float(wmax)

	# By default, the endpoint of the range is included
	wrange = np.linspace(wmin,wmax,n)

	# Generate a list of vdets for this refinement level
	vdetlist = [0.0 for k in range(0,n)]

	j = 0
	for w in wrange:		
		# Advance the composition to w
		computex(w)
		
		# Set the label for the torch output
		torchoutlabel = str(w)

		writetolog('starting torch runs...')
		# Find the overdriven velocity at w
		prevw = w
		if (j > 0):
			prevw = wrange[j-1]
		vdetopt = findoverdriven(vdetlist,j,prevw,w)
		foundvdet(xall['c12'],vdetopt)	
		vdetlist[j] = vdetopt

		# Archive torch log files, etc.
		tidytorch(w)

		# Write to log file
		xk = xall.keys()
		fsr.write(str(w) + ' ')
		fsr.write(str(vdetopt) + ' ')
		for k in xk:
			fsr.write(str(xall[k]) + ' ')
		fsr.write('\n')
		
		writetolog('Completed w: ' + str(w))
		j = j + 1
	return vdetopt


def testvdetsonic(vdet):
	writefile(vdet)
	runtorch()
	return geterrorstate()

def writetolog(s):
	global mainlog
	print s
	mainlog.write(s + '\n')
	

def findoverdriven(vdetlist,vdetindx,prevw,thisw):
	global vdettol
	global vdetopt
	global xall
	global roughvit
	global refpoints
	global currentrefine

	vdetstep = 0.0

	# Compute initial step:
	if (vdetindx >= 2):
		vdetstep = abs(vdetlist[vdetindx-1] - vdetlist[vdetindx-2])

	if (vdetstep < vdettol):
		vdetstep = vdettol

	hnan = hasnan(testvdetsonic(vdetopt))

	# Get minimally overdriven vdet...
	# Raise/lower vdet to get new upper/lower limit depending on hnan
	if hnan:
		writetolog('vdetopt: ' + str(vdetopt) + ' is sonic.')
		vdetlo = vdetopt
	else:
		writetolog('vdetopt: ' + str(vdetopt) + ' is not sonic.')
		vdethi = vdetopt
	raisefrac = 2.0
	for k in range(0,roughvit):
		if hnan:
			vdetk = vdetopt+vdetstep
		else:
			vdetk = vdetopt-vdetstep
		errstate = testvdetsonic(vdetk)
		if (k==roughvit-1):
			# Allowed iterations exceeded: need to refine
                        writetolog('Refining...')
			currentrefine = currentrefine + 1
                        thisvd = scanrange(prevw,thisw,refpoints)
			currentrefine = currentrefine - 1
			return thisvd
		elif hascriticalerror(errstate):
			# critical error, lower step
			# Note that the lowering fraction should be 
                        # less than 1 over the raising fraction
                        # since this will aid convergence instead of
                        # a forever alternating state.
			writetolog('Lowering step...')
			vdetstep = 0.3*vdetstep
			raisefrac = 1.0+(raisefrac-1.0)*0.7
			continue
		elif hasnan(errstate):
			# no critical error but still sonic, raise step
			writetolog('Raising step...')
			vdetstep = raisefrac*vdetstep
		else:
			# no critical error and not sonic, subdivide interval
			if hnan:
				vdethi = vdetk
			else:
				vdetlo = vdetk
			return subdividevdet(vdethi,vdetlo)
				
def subdividevdet(vdethi,vdetlo):	
	global maxvit
	global vdettol

	# Iterate maxvit times or until tolerance is reached
	domain = vdethi-vdetlo
	for j in range(0,maxvit):
		vdetj = vdetlo + 0.5*domain
		errstate = testvdetsonic(vdetj)
		if hasnan(errstate):
			vdetlo = vdetj
			domain = vdethi - vdetlo
		else:
			if (abs(vdetj-vdetlo) < vdettol):
				writetolog('Returning from subdivide, iteration: ' + str(j))
				writetolog('vdetlo is: ' + str(vdetlo))
				writetolog('vdethi is: ' + str(vdethi))
				writetolog('vdetj is: ' + str(vdetj))
				writetolog('domain is: ' + str(domain))
				return vdetj
			else:
				vdethi = vdetj
				domain = vdethi - vdetlo
		if domain < vdettol:
			return vdethi
	writetolog('Insufficient iterations maxvit to get required tolerance!')
	return vdethi
			
def foundvdet(xc12,vdet):
	writetolog('xc12: ' + str(xc12))
	writetolog('Found minimally overdriven vdet: ' + str(vdet))
	
def runtorch():
	subprocess.call('./torch < intorch.txt > foo_log.dat',shell=True)

def tidytorch(w):
	os.system('mv foo_log.dat foo_log_srw_' + str(w) + '.dat')
	os.system('mv intorch.txt intorch_' + str(w) + '.txt')

def hasnan(errstate):
	return errstate[0]

def hascriticalerror(errstate):
	return errstate[1]

def geterrorstate():
	flog = open('foo_log.dat','r')
	retval = [False, True]
	for l in flog:
		if (l.find('stepsize: nan error encountered')!=-1):
			# NaN error
			retval[0] = True
		if (l.find('STOP normal termination')!=-1):
			# Normal termination, no critical error
			retval[1] = False
	flog.close()
	return retval

def computex(w):
	# w normalized to wmax
	global xini
	global xall
	global xfin

	kall = xall.keys()
	for k in kall:
		if not k in xini:
			xall[k] = 0.0
		else:
			xall[k] = xini[k] + (xfin[k] - xini[k])*w
					
def writefile(vdet):
	global xall
	global torchoutlabel

	vdet = float(vdet)

	fin = open('intorch.txt','w')
	fin.write("8\n0\n")
	fin.write("1 0 0 0 0 0 0 1 1 1e30\n")
	fin.write("1\n0\n")
	fin.write("1e3 4e8 1e7\n")
	fin.write("3\n")

	strx = (str(xall['n']) + " " + str(xall['h1']) + " " + str(xall['he4']) + " " + 
		str(xall['c12']) + " " + str(xall['c13']) + " " + str(xall['n14']) + 
		" " + str(xall['o16']) + " " + str(xall['ne20']) + " " + 
		str(xall['ne22']) + " " + 
		str(xall['na23']) + " " + str(xall['mg24']) + " " +
		str(xall['si28']) + " " + str(xall['fe52']) + 
		" " + str(xall['fe54']) + " " + str(xall['fe56']) + " " + 
		str(xall['ni56']))

	fin.write(strx + '\n')
	fin.write("xc12_" + torchoutlabel + "_\n")
	fin.write(str(vdet) + '\n')
	fin.write("0")	
	fin.close()

# Call main
main()
