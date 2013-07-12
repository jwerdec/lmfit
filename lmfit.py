#!/usr/bin/env python
#
#   Requirements:
#   numpy, scipy.optimize, matplotlib, multiprocessing
#
#   Python v2.5 or later (not compatible with Python 3)
#
#   Version History:
#
#   v0.1 2013-05-21: First more or less fully functional version
#   v0.2 2013-06-16: Clean up and restructuring of the code
#   v0.3 2013-07-10: Bugfixes, New Bootstrapping function with
#                    multiprocessing support

from __future__ import division
import numpy as np
from numpy import sqrt, array, sum
import matplotlib.pylab as plt
from matplotlib.mlab import normpdf
import matplotlib.gridspec as gs
from scipy.optimize import leastsq
from sys import stderr
from traceback import print_exc
from multiprocessing import Process, JoinableQueue, Queue
plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

class lmfit(object):
    """
    Class providing access to Scipy's leastsq function for function fitting
    employing the Levenberg-Marqurdt algorithm.
    
    Resulting fitting parameters can be accessed via self.pvec or individually
    via self.pvec['name of parameter'].
    """

    # CONSTRUCTOR
    
    def __init__(self, func, xdata, ydata, p0, yerror=None, lm_options={},\
                     verbose=True, plot=False, plot_options={}):
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
        # Check input
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
        
        self.__x = xdata
        self.__y = ydata
        self.__yerror = yerror
        self.__ndf = self.__reclength - len(p0) - 1
        self.__func = func
        params, self.__pNames = self.__getParams(p0)
        self.__p0 = dict(zip(self.__pNames, params))
        self.fit(params, lm_options=lm_options, verbose=verbose, plot=plot,\
                     plot_options = plot_options)

    # PRIVATE METHODS

    def __getParams(self, p0):
        VarNames = self.__getVarNames(self.__func)
        i = 1
        if VarNames[0] == 'self':
            i += 1
        pNames = VarNames[i:]
        if type(p0) == type({}):
            p0List = self.__pDictToList(p0, pNames)
            return p0List, pNames
        else:
            return p0, pNames
            
    def __getVarNames(self, func):
        return func.func_code.co_varnames

    def __pDictToList(self, pdict, PNames):
        PList = []
        for PName in PNames:
            PList.append(pdict[PName])
        return PList

    def __ToMinimize(self, params):
        pass

    def __Residuals(self, params):
        return(self.__y - self.__func(self.__x,  *params))

    def __WeightedResiduals(self, params):
        return(self.__y - self.__func(self.__x, *params)/self.__yerror)

    # SPECIAL PYTHON METHODS

    def __call__(self, x):
    	"""
    	Evaluates the test function at x with the current set of parameters
        """
        return self.__func(x, *self.__pfinal)

    # PUBLIC METHODS

    @property
    def P(self):
        return self.__pfinalDict

    @property
    def StdDev(self):
        return self.__StdDev

    def fit(self, p0, lm_options={}, verbose=True, plot=False, plot_options={}):
        """
  	Carries out the least squares optimization.
        """
        params, self.__pNames = self.__getParams(p0)
        self.__p0 = dict(zip(self.__pNames, params))
        np.seterr(divide='raise')
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

        self.__pfinalDict = dict(zip(self.__pNames, self.__pfinal))
        Chi2 = sum(self(self.__x)**2)
        VarRes = Chi2 / self.__ndf
        RMSChi2 = sqrt(VarRes)
        CovMatrix = covx * VarRes
        StdDev = dict(zip(self.__pNames, [sqrt(i) for i in np.diag(CovMatrix)]))
        Residuals = self.__Residuals(self.__pfinal)
        self.__StdDev = StdDev
        self.__Res = Residuals
        self.__results = {\
            'Parameters': self.__pfinalDict, 'CovMatr': CovMatrix, 'Chi2': Chi2,\
            'VarRes': VarRes, 'RMSChi2': RMSChi2, 'StdDev': StdDev, \
            'Residuals': Residuals, 'nfev': infodict['nfev'], 'fvec': infodict['fvec'],\
            'MINPACKMsg': msg}
        if verbose:
        	self.report()
        if plot:
        	self.plot(**plot_options)

    @property 
    def CovMatrix(self):
        return self.__results['CovMatr']
    
    @property
    def Chi2(self):
        return self.__results['Chi2']

    @property
    def RMSChi2(self):
        return self.__results['RMSChi2']

    @property
    def Residuals(self):
        return self.__results['Residuals']

    @property
    def full_results(self):
        return self.__results

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
    	cols = 2
    	height_ratios=[2]
    	width_ratios=[1,1]
    	if residuals:
    		rows += 1
    		height_ratios.append(1)
        if lagplot:
            rows += 2
            height_ratios.append(1)
            height_ratios.append(1)
        elif histogramm or acf:
            rows += 1
            height_ratios.append(1)
    	# set geometry
    	grid = gs.GridSpec(rows, cols, width_ratios=width_ratios,\
    		 height_ratios=height_ratios)
    	# main plot	 
    	fig = plt.figure(figsize=(16,10), dpi=100)
    	fitplot = plt.subplot(grid[0, :])
    	fitplot.set_xlabel(r'$x$')
    	fitplot.set_ylabel(r'$y$')
    	row = 1
    	if self.__yerror != None:
            fitplot.errorbar(self.__x, self.__y, self.__yerror,\
                             fmt='o')
        else:
            fitplot.plot(self.__x, self.__y, 'o', label='Data')
        x = np.linspace(self.__x[0], self.__x[-1], 1000)
        fitplot.plot(x, self(x), 'r-', label='Fit')
        fitplot.legend(loc='best', numpoints=1)
        # additional plots
        col = 0
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
        	lags, corrs, line, xaxis =\
                    acfplot.acorr(self.__Res, maxlags=None)
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
        grid.tight_layout(fig)
        self.fig = fig
        plt.show(fig)
        
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

    def __fit_worker(self, q_in, q_out):
        while True:
            y = q_in.get()
            if y is None:
                q_in.task_done()
                break
            ToMinimize = lambda params: y - self.__func(self.__x, *params)
            pfinal, covx, infodict, msg, ier =\
                leastsq(func=ToMinimize, x0=self.__pfinal, full_output=1)
            pfinalDict = dict(zip(self.__pNames, pfinal))
            q_out.put(pfinalDict)
            q_in.task_done()
        return

    def bootstrap(self, n=500, n_cpu=1, plot=False):
        """
        BETA!
	Resamples the residuals and fits the data to the fitted function + resampled residuals. Then calculates the statistics for the fitting parameters.
	n: Number of resamplings
        n_cpu: Number of processes used by this method
        Returns:
            Mean: Dictionary of Mean values found for all fit parameters
            StdDev: Dictionary of Standard Deviations for all fit parameters
            outlist: A list with dictionaries of fit parameters from all fits 
	"""    

        def __fit_worker(self, q_in, q_out):
            """
            Defines what is done in the multiprocessing step
            """
            while True:        # watch out for new input in a loop
                y = q_in.get() # get input from queue
                if y is None:  # break the loop, when receiving None-type
                    q_in.task_done()
                    break
                ToMinimize = lambda params: y - self.__func(self.__x, *params)
                pfinal, covx, infodict, msg, ier =\
                    leastsq(func=ToMinimize, x0=self.__pfinal, full_output=1)
                pfinalDict = dict(zip(self.__pNames, pfinal))
                q_out.put(pfinalDict) #pass result to the output queue
                q_in.task_done()
            return

        # introducing the queues
        q_in = JoinableQueue()
        q_out = Queue()
        # set up the processes
        procs = []
        for i in range(n_cpu):
            p = Process(target=self.__fit_worker, args=(q_in,q_out))
            p.daemon = True
            p.start()
            procs.append(p)
        outlist = [i for i in range(n)]
        # set up the jobs to do and put them into the queue
        for i in outlist:
            NewY = self(self.__x) + np.random.permutation(self.__Res)
            q_in.put(NewY)
        for i in range(n_cpu): # tell processes to break
            q_in.put(None)
        q_in.join()           # wait for all processes to finish
        for i in outlist:     # get the results
            outlist[i] = q_out.get()
        Results = {}
        for varname in self.__pNames:
            Results[varname] = [outdict[varname] for outdict in outlist]
        # Calculate Mean and Standard Deviation
        Mean = {}
        StdDev = {}
        for varname in Results.keys():
            Mean[varname] = np.array(Results[varname]).mean()
            StdDev[varname] = np.array(Results[varname]).std()
        if plot: # plot fits if requested
            plt.plot(self.__x, self.__y, 'bo')
            plt.title(r'Bootstrapping Fits')
            x = np.linspace(self.__x[0], self.__x[-1], 1000)
            for finalDict in outlist:
                plt.plot(x, self.__func(x, **finalDict))
            plt.show()
        return Mean, StdDev, outlist
        

# TESTCODE
if __name__ == '__main__':
    from numpy import exp, linspace, array
    from numpy.random import normal
    # Signal generating function
    sig = lambda x, alpha, x0: x**3/(x0**3) * exp(-((x-x0)/alpha)**2) + normal(0,0.1,len(x))
    # Test function
    testfunc = lambda x, alpha, x0: x**3.0/(x0**3.0) * exp(-((x-x0)/alpha)**2)
    # Creating a synthetic dataset
    x = array(linspace(0, 100, 1000))
    y = sig(x=x, alpha=9.67, x0=18.47)
    # Fitting the data with the testfunction by creating an instance of 
    # the lmfit class
    testfit = lmfit(testfunc, xdata=x, ydata=y, p0={'x0':20, 'alpha':10}, plot=True)
    Mean, StdDev, outlist = testfit.bootstrap(500, 2, plot=True)
    print Mean, StdDev
