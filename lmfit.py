from __future__ import division
from sys import stderr
import sys
from traceback import print_exc

import numpy as np
from numpy import sqrt, array, sum
import matplotlib.pylab as plt
from matplotlib.mlab import normpdf
import matplotlib.gridspec as gs
from scipy.optimize import leastsq

class LMFit(object):
    r"""
    Class handling non-linear least squares fitting of 2d datasets.

    Based on scipy's leastsq function lmfit implements the
    Levenberg-Marquardt-Algorithm provided by the Fortran MINPACK library.

    Parameters
    ----------
    func : function-type
        Testfunction to be fitted.
    xdata, ydata: array-like
        Datasets with equal dimensions.
    p0 : dict or list
        Set of initial parameters. When passing p0 as a list the ordering
        of the parameters must be the same as in the function definition.
        So for func = lambda x, a, b: a*x + b either is possible:
        p0={'a':1, 'b': 2} or p0=[1, 2].
    yerror : array-like, optional
        Weights for individual data points. Must have the same dimensions as
        x/ydata arrays.
    lm_options : dict, optional
        Options passed to scipy.optimize.leastsq. Cf. scipy reference for 
        possible options.
    verbose, plot : bool, optional
        Toggle verbose output (default: True) and plot window (default: False).
    plot_options : dict, optional
        Options passed to this classes plot method.

    Attributes
    ----------
    xdata, ydata
    func
    P 
    StdDev
    CovMatr
    Chi2
    RMSChi2
    Residuals
    full_results
    fig : class instance
        Instance of matplotlib's figure class. Only available after plotting.

    See Also
    --------
    scipy.optimize.leastsq : Wrapper for MINPACK's fit functions.

    Notes
    -----
    """
    
    def __init__(self, func, xdata, ydata, p0, yerror=None, lm_options={},\
                     verbose=True, plot=False, plot_options={}):
        # Check input
        self.__reclength = len(xdata)
        if self.__reclength != len(ydata):
            stderr.write("\nERROR: Inconsistent number of data points in xdata "
                         "and ydata!\n")
            return		
        if yerror == None:
            self.__ToMinimize = self.__Residuals
        elif self.__reclength == len(yerror):
            self.__ToMinimize = self.__WeightedResiduals
        else:
            stderr.write("\nERROR: Inconsistent number of data points in data "
                         "and yerror!\n")
            return
        # Write variables
        self.__x = xdata
        self.__y = ydata
        self.__yerror = yerror
        self.__ndf = self.__reclength - len(p0) - 1
        self.__func = func
        params, self.__pNames = self.__getParams(p0)
        self.__p0 = dict(zip(self.__pNames, params))
        self.fit(params, lm_options=lm_options, verbose=verbose, plot=plot,\
                     plot_options = plot_options)

    def __getParams(self, p0):
        r"""
        Get the names of the parameters from p0.
        """
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
        r"""
        Get the name of the arguments of the function func.
        """
        return func.func_code.co_varnames[:func.func_code.co_argcount]

    def __pDictToList(self, pdict, PNames):
        r"""
        Puts the parameters in pdict into list in the ordering given by PNames. 
        """
        PList = []
        for PName in PNames:
            PList.append(pdict[PName])
        return PList

    def __ToMinimize(self, params):
        r"""
        Dummy method which is replaced by __Residuals or __WightedResiduals on
        contruction.
        """
        pass

    def __Residuals(self, params):
        r"""
        Defines the function whos sum of squares is to be minimized.
        """
        return(self.__y - self.__func(self.__x,  *params))

    def __WeightedResiduals(self, params):
        r"""
        Defines the function whos sum of squares is to be minimized.
        """
        return(self.__y - self.__func(self.__x, *params))/self.__yerror

    def __call__(self, x):
    	"""
    	Evaluates the testfunction at x with the current set of parameters
        """
        return self.__func(x, *self.__pfinal)

    @property
    def P(self):
        r"""
        Dictionary containing the resulting fit parameters.
        """
        return self.__pfinalDict

    @property
    def StdDev(self): 
        r"""
        Dictionary containing the standard deviations of the resulting fit
        parameters.
        """
        return self.__StdDev

    def fit(self, p0, lm_options={}, verbose=True, plot=False, plot_options={}):
        r"""
  	Carries out the non-linear fit.

        Parameters
        ----------
        p0 : dict or list
            Set of initial parameters. When passing p0 as a list the ordering
            of the parameters must be the same as in the function definition.
            So for func = lambda x, a, b: a*x + b either is possible:
            p0={'a':1, 'b': 2} or p0=[1, 2].
        yerror : array-like, optional
            Weights for individual data points. Must have the same dimensions as
            x/ydata arrays.
        lm_options : dict, optional
            Options passed to scipy.optimize.leastsq. Cf. scipy reference for 
            possible options.
        verbose, plot : bool, optional
            Toggle verbose output (default: True) and plot window
            (default: False).
        plot_options : dict, optional
            Options passed to this classes plot method.
        
        Raises
        ------
        FloatingPointError
            If the testfunction can be evaluated given the initial parameters.
        Exception
            If scipy.optimize.leastsq fails without raising an own exception.
        """
        params, self.__pNames = self.__getParams(p0)
        self.__p0 = dict(zip(self.__pNames, params))
        np.seterr(divide='raise') # make numpy raise an exception
        try:
            self.__func(self.__x, *params)
        except FloatingPointError as FPE:
            stderr.write("\nERROR: Testfunction could not be evaluated using "
                         "the given initial parameters!\n(%s)\n\n" %FPE) 
            print_exc(limit=2)
            return
        try:
            self.__pfinal, covx, infodict, msg, ier =\
                leastsq(func=self.__ToMinimize, x0=params, full_output=1,
                        **lm_options)
        except:
            raise Exception("An unknown error has occured in the fitting "
                            "process!")
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
        self.__results = {
            'Parameters': self.__pfinalDict, 'CovMatr': CovMatrix,
            'Chi2': Chi2, 'VarRes': VarRes, 'RMSChi2': RMSChi2,
            'StdDev': StdDev, 'Residuals': Residuals, 'nfev': infodict['nfev'],
            'fvec': infodict['fvec'], 'MINPACKMsg': msg}
        if verbose:
        	self.report()
        if plot:
        	self.plot(**plot_options)
        	
    @property
    def xdata(self):
    	return self.__x
    	
    @property
    def ydata(self):
        return self.__y

    @property 
    def CovMatrix(self):
        r"""
        The Covariance Matrix.
        """
        return self.__results['CovMatr']
    
    @property
    def Chi2(self):
        r"""
        Value of Chi^2.
        """
        return self.__results['Chi2']

    @property
    def RMSChi2(self):
        r"""
        Root mean square value of Chi^2.
        """
        return self.__results['RMSChi2']

    @property
    def Residuals(self):
        r"""
        Array containing the Residuals.
        """
        return self.__results['Residuals']

    @property
    def full_results(self):
        r"""
        Dictionary with all results from the fit and additional information.
        """
        return self.__results

    @property
    def func(self):
        r"""
        The original testfunction.
        """
        return self.__func

    def plot(self, residuals=True, acf=True, lagplot=True, histogramm=True):
    	r"""
    	Creates a plot of the data and the test function using current
        parameters.

    	Parameters
        ----------
    	residuals : bool, optional	
            Plot the residuals
    	acf : bool, optional
	    Plot the autocorrelogramm
    	lagplot : bool, optional
            Show a lagplot
    	histogramm : bool, optional
            Plot histogramm of residuals 
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
    	r"""
    	Prints a report about the results of the last fitting procedure.
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

    def bootstrap(self, n=500, plot=False, full_output=False):
        r"""
        Performs a bootstrapping analysis of the Residuals.

        The Residuals are randomly resampled and superimposed on the fitted 
        testfunction. This artificial dataset is then fitted again and the 
        final parameters are stored. This is repeated n-times. In the end
        the mean values and standard deviations of the fit parameters from 
        all fits are calculated and returned.

        Parameters
        ----------
        n : int, optional
            Number of bootstrapping runs (default=500).
        plot : bool, optional
            Plot all fits (default=False). Be careful! Can be slow for large
            values of n.

        Returns
        -------
        Mean : dict
            Mean values of the fit parameters from all bootstrap fits.
        StdDev: dict
            Standard Deviations of the fit parameters determined from all
            bootstrap fits.
        outlist : dict
            List of dictionaries containing all final parameter sets from
            the bootstrap fits.
	"""    
        sys.stderr.write('Bootstrapping.')
        outlist = [i for i in range(n)]
        for i in outlist:
            sys.stderr.write('.')
            NewY = self(self.__x) + np.random.permutation(self.__Res)
            ToMinimize = lambda params: NewY - self.__func(self.__x, *params)
            pfinal, covx, infodict, msg, ier =\
                    leastsq(func=ToMinimize, x0=self.__pfinal, full_output=1)
            outlist[i] = dict(zip(self.__pNames, pfinal))
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
	if full_output:
        	return Mean, StdDev, outlist
	return Mean, StdDev
        
# TESTCODE
if __name__ == '__main__':
    from numpy import exp, linspace, array
    from numpy.random import normal
    # Signal generating function
    sig = lambda x, alpha, x0: x**3/(x0**3) * exp(-((x-x0)/alpha)**2)\
        + normal(0,0.1,len(x))
    # Test function
    testfunc = lambda x, alpha, x0: x**3.0/(x0**3.0) * exp(-((x-x0)/alpha)**2)
    # Creating a synthetic dataset
    x = array(linspace(0, 100, 1000))
    y = sig(x=x, alpha=9.67, x0=18.47)
    # Fitting the data with the testfunction by creating an instance of 
    # the lmfit class
    testfit = LMFit(testfunc, xdata=x, ydata=y, p0={'x0':20, 'alpha':10},
                    plot=True)
    Mean, StdDev, outlist = testfit.bootstrap(500, plot=True)
    print Mean, StdDev
