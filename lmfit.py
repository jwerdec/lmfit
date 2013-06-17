#!/usr/bin/env python
#
#   Requirements:
#
#   Python v2.5 or later (not compatible with Python 3)
#
#   Version History:
#
#   v0.1 2013-05-21: First more or less fully functional version
#   v0.2 2013-06-16: Clean up and restructuring of the code

from __future__ import division
import numpy as np
from numpy import sqrt, array, sum
import matplotlib.pylab as plt
from matplotlib.mlab import normpdf
import matplotlib.gridspec as gs
from scipy.optimize import leastsq
from sys import stderr
from traceback import print_exc
np.seterr(divide='raise')

class lmfit(object):
    """
    Class providing access to Scipy's leastsq function for function fitting
    employing the Levenberg-Marqurdt algorithm.
    
    Resulting fitting parameters can be accessed via self.pvec or individually
    via self.pvec['name of parameter'].
    """

    # CONSTRUCTOR
    
    def __init__(self, func, xdata, ydata, p0, yerror=None, lm_options={}, verbose=True, \
    		plot=False, plot_options={}):
        """
        Constructor of class lmfit.
        Parameters:
        func: 		  Pointer on the testfunction used to fit the data with
        xdata, ydata: Lists or 1D-Arrays of the same length containing the \\
        	data points
        yerror:  	  List or 1D-Array of the same length as ydata containing\\
        	weights to the individual ydata points. None (default) weights \\
        	every y value equally
        p0:			  Dictionary of the parameters in func to optimize
        lmoptions:	  Options passed to scipy.optimize.leastsq function, cf. \\
        	Scipy manual for available options
        verbose:	  Prints a full report on every fit (default: True)
        plot:		  Generates Plots to analyze the fit
        """
        # Checking input
        self.__reclength = len(xdata)
        if self.__reclength != len(ydata):
            stderr.write('ERROR: Inconsistent number of data points in xdata and ydata!\n')
            return
			
        if yerror == None:
            self.__ToMinimize = self.__Residuals
        elif self.__reclength == len(yerror):
            self.__ToMinimize = self.__WeightedResiduals
        else:
            stderr.write('ERROR: Inconsistent number of data points in data and yerror!\n')
            return
    
        self.__p0 = p0
        self.__func = func
        self.__x = xdata
        self.__y = ydata
        self.__yerror = yerror
        self.__ndf = self.__reclength - len(self.__p0) - 1
        self.fit(p0, lm_options, verbose, plot, plot_options)

    # PRIVATE METHODS

    def __ToMinimize(self, params):
        pass

    def __Residuals(self, params):
        return(self.__y - self.__func(self.__x, *params))

    def __WeightedResiduals(self, params):
        return(self.__y - self.__func(self.__x, *params)/self.__yerror)

    # SPECIAL PYTHON METHODS

    def __call__(self, x):
    	"""
    	Evaluates the test function at x with the current set of parameters
        """
        return self.__func(x, *self.__pfinal)

    # PUBLIC METHODS

    def getParameters(self):
        return self.__pfinalDict

    def getStdDev(self):
        return self.__StdDev

    P = property(getParameters)
    StdDev = property(getStdDev)

    def fit(self, p0, lm_options={}, verbose=True, plot=False, plot_options={}):
        """
  	Carries out the least squares optimization.
        """
        self.__results = None
        self.__p0 = p0
        self.__pnames = self.__p0.keys()    
        params = self.__p0.values()
        try:
            self.__func(self.__x, *params)
        except FloatingPointError as FPE:
            stderr.write('\nERROR: Testfunction could not be evaluated using the\
 given initial parameters!\n(%s)\n\n' %FPE) 
            print_exc(limit=2)
            return
        	
        try:
            self.__pfinal, covx, infodict, msg, ier =\
                leastsq(func=self.__ToMinimize, x0=params, full_output=1, **lm_options)
        except:
            raise Exception("An unknown error has occured in the fitting process!")
            return

        
        if covx == None:
            print """
Warning: Fit did not converge properly!

Message provided by MINPACK:
%s
""" % msg
            return

        Chi2 = sum(self(self.__x)**2)
        VarRes = Chi2 / self.__ndf
        RMSChi2 = sqrt(VarRes)
        CovMatrix = covx * VarRes
        StdDev = dict(zip(self.__pnames, [sqrt(i) for i in np.diag(CovMatrix)]))
        Residuals = self.__Residuals(self.__pfinal)
        self.__pfinalDict = dict(zip(self.__pnames, self.__pfinal))
        self.__StdDev = StdDev
        self.__Res = Residuals
        self.__results = {\
            'Parameters': self.__pfinal, 'CovMatr': CovMatrix, 'Chi2': Chi2,\
            'VarRes': VarRes, 'RMSChi2': RMSChi2, 'StdDev': StdDev, \
            'Residuals': Residuals, 'nfev': infodict['nfev'], 'fvec': infodict['fvec'],\
            'MINPACKMsg': msg}
        if verbose:
        	self.report()
        if plot:
        	self.plot()

    def plot(self, residuals=True, acf=True, lagplot=True, histogramm=True):
    	"""
    	Creates a plot of the data and the test function using current parameters.
    	Options:
    		residuals  (bool):	Plot the residuals
    		acf        (bool):		Plot the autocorrelogramm
    		lagplot    (bool):	Show a lagplot
    		histogramm (bool): Plot histogramm of residuals 
    	"""
    	# determine geometry
    	rows = 1
    	cols = 1
    	height_ratios=[2]
    	width_ratios=[1]
    	if residuals:
    		rows += 1
    		height_ratios.append(1)
    	if acf:
    		rows += 1
    		height_ratios.append(1)
    	if lagplot:
    		cols = 2
    		width_ratios=[1,1]
    	if histogramm:
    		if acf and lagplot:
    			rows += 1
    			height_ratios.append(1)
    		cols = 2
    		width_ratios=[1,1]
    	# set geometry
    	grid = gs.GridSpec(rows, cols, width_ratios=width_ratios,\
    		 height_ratios=height_ratios)
    	# main plot	 
    	fig = plt.figure()
    	fitplot = plt.subplot(grid[0, :])
    	fitplot.set_xlabel(r'$x$')
    	fitplot.set_ylabel(r'$y$')
    	row = 1
    	if self.__yerror != None:
            fitplot.errorbar(self.__x, self.__y, self.__yerror,\
                             fmt='o')
        else:
            fitplot.plot(self.__x, self.__y, 'o', label='Data')
        x = np.linspace(self.__x[0], self.__x[len(self.__x)-1],\
                         len(self.__x))
        fitplot.plot(x, self(x), 'r-', label='Fit')
        fitplot.legend(loc='best')
        # additional plots
    	if residuals:
        	residualplot = plt.subplot(grid[row, :])
        	row += 1
        	residualplot.plot(self.__x, self.__Res, 'go')
        	residualplot.set_title(u'Plot of Residuals')
        	residualplot.set_xlabel(r'$x$')
        	residualplot.set_ylabel(r'Residuals')
    	if acf:
    		col = 0
    		acfplot = plt.subplot(grid[row, col])
    		col = 1
        	lags, corrs, line, xaxis = acfplot.acorr(self.__Res, maxlags=None)
        	acfplot.set_xlim((-(len(lags)-1)/200, (len(lags)-1)/2))
        	acfplot.set_title(u'Correlogram of Residuals')
        	acfplot.set_xlabel(u'Lag')	
        	acfplot.set_ylabel(u'Autocorrelation')
        if lagplot:
        	lagplot = plt.subplot(grid[row:, col])
        	if col == 1:
        		col = 0
        	row += 1
        	lagplot.plot(self.__Res[1:], self.__Res[:-1], 'g.')
        	lagplot.set_title(u'Lag Plot of Residuals')	
        	lagplot.set_xlabel(r'$Y_{i-1}$')	
        	lagplot.set_ylabel(r'$Y_i$')
       	if histogramm:
        	hist = plt.subplot(grid[row, col])
        	n, bins, patches = plt.hist(self.__Res, bins=20, normed=1,\
                                                histtype='stepfilled')
        	data = array(self.__Res)
    		mu = data.mean()
    		sigma = data.std()
    		gauss = normpdf(bins,mu,sigma)
    		plt.plot(bins,gauss,'ro-')
        	hist.set_title(u'Histogramm')	
        	hist.set_xlabel(r'')	
        	hist.set_ylabel(r'')
        grid.tight_layout(fig, h_pad=-1)
        plt.show()
        return(fig)
        
    def report(self):
    	"""
    	Prints a report about the results of the last fitting procedure
    	"""
        if self.__results == None:
            print 'No results to report!'
            return
        pstring = ''
        for item in self.__pfinalDict:
            pstring += "\n%s = %f +/- %f"\
                %(item, self.__pfinalDict[item], self.__StdDev[item])
        print """
===========================================
Report:
-------------------------------------------
Initial set of parameters:
%s
Number of degrees of freedom (ndf): %d

Results:
Number of function evaluations: %d
      Sum of residuals (Chi^2): %f
 Variance of Chi^2 (Chi^2/ndf): %f
RMS of Chi^2 (sqrt(Chi^2/ndf)): %f

Covariance Matrix:
%s

Final set of parameters: %s
===========================================
""" % (self.__p0, self.__ndf, self.__results['nfev'], self.__results['Chi2'], \
           self.__results['VarRes'], self.__results['RMSChi2'],\
           self.__results['CovMatr'], pstring)

    def bootstrap(self, n=20):
        """
        BETA!
		Resamples the residuals and fits the data to the fitted function + 
		resampled residuals. Then calculates the statistics for the fitting 
		parameters.
		n: Number of resamplings
		"""
        self.bootstrapfits = [i for i in range(n)]
        for i in self.bootstrapfits:
            NewY = self(self.__x) + np.random.permutation(self.__Res)
            self.bootstrapfits[i] = lmfit(self.__func, self.__x, NewY,\
                                              p0=self.__pfinalDict, verbose=False)
        for item in self.__pfinalDict:
            mean = array([fit.__pfinalDict[item] for fit in self.bootstrapfits]).mean()
            stddev = array([fit.__pfinalDict[item] for fit in \
                self.bootstrapfits]).std()
            print "%s = %f +/- %f" %(item, mean, stddev)

# TESTCODE
if __name__ == '__main__':
    from numpy import exp, linspace, array
    from numpy.random import normal
    # Signal generating function
    sig = lambda x, alpha, x0: x**3/(x0**3) * exp(-((x-x0)/alpha)**2) + normal(0,0.1,len(x))
    # Test function
    testfunc = lambda x, alpha, x0: x**3/(x0**3) * exp(-((x-x0)/alpha)**2)
    # Creating a synthetic dataset
    x = array(linspace(0, 100, 1000))
    y = sig(x=x, alpha=9.67, x0=18.47)
    # Fitting the data with the testfunction by creating an instance of 
    # the lmfit class
    testfit = lmfit(testfunc, xdata=x, ydata=y, p0={'x0':20, 'alpha':10}, plot=True)
    # Bootstrap option: Remove the hash in the next line to perform a bootstrap analyses
    #testfit.bootstrap(20)
