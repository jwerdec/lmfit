#!/usr/bin/env python
#
#   Requirements:
#
#   Python v2.5 or later (not compatible with Python 3)
#
#   Version History:
#
#   v0.1 2013-05-21: First more or less fully functional version

import numpy as np
from numpy import sqrt, array
import matplotlib.pylab as plt
from matplotlib.mlab import normpdf
import matplotlib.gridspec as gs
from scipy.optimize import leastsq

class lmfit(object):
    """
    Class providing access to Scipy's leastsq function for function fitting
    employing the Levenberg-Marqurdt algorithm.
    
    Resulting fitting parameters can be accessed via self.pvec or individually
    via self.pvec['name of parameter']
    """
    def __init__(self, func, xdata, ydata, yerror=None, p0={}, lmoptions={}, verbose=True, plot=False):
        """
        Constructor of class lmfit.
        Parameters:
        func: 		  Pointer on the testfunction used to fit the data with
        xdata, ydata: Lists or 1D-Arrays of the same length containing the \\
        	data points
        yerror:  	  List or 1D-Array of the same length as ydata containing\\
        	weights to the individual ydata points. None (default value) weights \\
        	every y value equally
        p0:			  Dictionary of the parameters in func to optimize
        lmoptions:	  Options passed to scipy.optimize.leastsq function, cf. \\
        	Scipy manual for available options
        verbose:	  Prints a full report on every fit (default: True)
        plot:		  Generates Plots to analyze the fit
        """
        self.__lmoptions = lmoptions
        self.__verbose = verbose
        self.__plot = plot
        if p0 == {}:
            print "Error in fit.__init__(): No initial parameters p0 given."
            return(0)  
        self.__pnames = p0.keys()
        self.pvec = p0.values()
        self.__p0 = p0
        if len(xdata) != len(ydata):
            print "Error in fit.__init__(): Inconsistent number of data points in xdata and ydata."
            return(0)
        self.__func = func
        self.__xdata = xdata
        self.__ydata = ydata
        self.__yerror = yerror
        if yerror == None:
            self.__weight = array([1 for i in xdata])
        else:
            self.__weight = yerror
        self.fit()

    def __chi2(self, pvec):
        """
        Returns the value of chi^2.
        """
        chi2 = 0
        params = dict(zip(self.__pnames, pvec))
        i = 0
        for x in self.__xdata:
            chi2 +=\
                ((self.__ydata[i] - self.__func(x, **params))/ \
                     self.__weight[i])**2
            i += 1
        return chi2

    def __pdict(self, pvec):
    	"""
    	Creates a dictionary containing the fitting parameters
    	"""
        return dict(zip(self.__pnames, pvec))

    def __wr(self, pvec):
        """
        Defines the set of equations whose sum of squares is minimized
        through the least squares algorithm.
        """
        return ((self.__ydata - self.__func(self.__xdata,\
                                                **self.__pdict(pvec))) / \
                    self.__weight)

    def __residualvariance(self):
        """
        Calculates the variance of residuals (chi^2 / ndf) (reduced chi^2).
        """
        return (self.__chi2(self.pvec) / self.__ndf())

    def __ndf(self):
        """
        Calculates the number of degree of freedom (N - n) where N the
        number of data points and n is the number of parameters.
        """
        return (len(self.__xdata) - len(self.pvec))

    def fit(self):
        """
  		Carries out the least squares optimization.
        """
        lm = leastsq(func=self.__wr,\
                         x0=self.pvec,\
                         full_output=1,\
                         **self.__lmoptions)
        if 4 < lm[4] < 1:
            print "Error: No solution found. Error message provided by \
            	 scipy.optimize.leastsq:"
            print lm[3]
            return(0)
        self.pvec = lm[0]
        self.covmatr = lm[1] * self.__residualvariance()
        self.chi2 = self.__chi2(self.pvec)
        self.stddev =\
            dict(zip(self.__pnames,\
                         [sqrt(i) for i in np.diag(self.covmatr)]))
        self.parameters = dict(zip(self.__pnames, self.pvec))
        self.residuals = self.__residuals()
        if self.__verbose:
        	self.report()
        if self.__plot:
        	self.plot()

    def __residuals(self):
    	"""
    	Calculates the residuals
    	"""
        residuals = [None for x in self.__xdata]
        i = 0
        for x in self.__xdata:
            residuals[i] = self.__ydata[i] - self(x)
            i += 1
        return residuals

    def __call__(self, x):
    	"""
    	Evaluates the test function at x with the current set of parameters
        """
        return self.__func(x, **self.parameters)

    def plot(self, residuals=True, acf=True, lagplot=True, histogramm=True):
    	"""
    	Creates a plot of the data and the test function using current parameters.
    	Options:
    		residuals:	Plot the residuals
    		acf:		Plot the autocorrelogramm
    		lagplot:	Show a lagplot
    		histogramm: Plot histogramm of residuals 
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
            fitplot.errorbar(self.__xdata, self.__ydata, self.__yerror,\
                             fmt='o')
        else:
            fitplot.plot(self.__xdata, self.__ydata, 'o', label='Data')
        x = np.linspace(self.__xdata[0], self.__xdata[len(self.__xdata)-1],\
                         len(self.__xdata))
        fitplot.plot(x, self(x), 'r-', label='Fit')
        fitplot.legend(loc='best')
        # additional plots
    	if residuals:
        	residualplot = plt.subplot(grid[row, :])
        	row += 1
        	residualplot.plot(self.__xdata, self.residuals, 'go')
        	residualplot.set_title(u'Plot of Residuals')
        	residualplot.set_xlabel(r'$x$')
        	residualplot.set_ylabel(r'Residuals')
    	if acf:
    		col = 0
    		acfplot = plt.subplot(grid[row, col])
    		col = 1
        	lags, corrs, line, xaxis = acfplot.acorr(self.residuals, maxlags=None)
        	acfplot.set_xlim((-(len(lags)-1)/200, (len(lags)-1)/2))
        	acfplot.set_title(u'Correlogram of Residuals')
        	acfplot.set_xlabel(u'Lag')	
        	acfplot.set_ylabel(u'Autocorrelation')
        if lagplot:
        	lagplot = plt.subplot(grid[row:, col])
        	if col == 1:
        		col = 0
        	row += 1
        	lagplot.plot(self.residuals[1:], self.residuals[:-1], 'g.')
        	lagplot.set_title(u'Lag Plot of Residuals')	
        	lagplot.set_xlabel(r'$Y_{i-1}$')	
        	lagplot.set_ylabel(r'$Y_i$')
       	if histogramm:
        	hist = plt.subplot(grid[row, col])
        	n, bins, patches = plt.hist(self.residuals, bins=20, normed=1, histtype='stepfilled')
        	data = array(self.residuals)
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
    	# Really not the most elegant way to do this...
        print "============================================"
        print "Report:"
        print "--------------------------------------------"
        print "Initial set of parameters:"
        print self.__p0
        print "Number of degrees of freedom (ndf): %d" % self.__ndf()
        print ""
        print "Results:"
        print ""
        print "      Sum of residuals (Chi^2): %f" % self.chi2
        print " Variance of Chi^2 (Chi^2/ndf): %f"\
            % (self.chi2/self.__ndf())
        print "RMS of Chi^2 (sqrt(Chi^2/ndf)): %f"\
            % (sqrt(self.chi2/self.__ndf()))
        print ""
        print "Covariance Matrix:"
        print self.covmatr
        print ""
        print "Final set of parameters:"
        for item in self.parameters:
            print "%s = %f +/- %f" %(item, self.parameters[item],\
                                         self.stddev[item])
        print "============================================"
        
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
            NewY = self(self.__xdata) + np.random.permutation(self.residuals)
            self.bootstrapfits[i] = lmfit(self.__func, self.__xdata, NewY, p0=self.parameters, \
	       		verbose=False)
        for item in self.parameters:
            mean = array([fit.parameters[item] for fit in self.bootstrapfits]).mean()
            stddev = array([fit.parameters[item] for fit in \
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
	testfit = lmfit(testfunc, xdata=x, ydata=y, p0={'x0':20, 'alpha':10})
	# Plot the results
	testfit.plot() 
