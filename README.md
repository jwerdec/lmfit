lmfit
=====

Provides a class handling non-linear least squares fitting of 2d datasets 
(i.e. functions of one independent variable) in a convenient way. 
Based on scipy's leastsq function it implements the well known 
Levenberg-Marquardt-Algorithm provided by the Fortran MINPACK library.
The lmfit class has an intuitive user interface and the presentation 
of the fit result is given in a gnuplot inspired way.
Also included are some options and plotting features for instant
analysis of the fit result, e.g. Autocorrelogram, Lag Plot, Plot of
Residuals and Histogramm. 
One more special feature is the bootstrap method which will resample
the residuals randomly and superimpose them upon the fitted function 
and then refitting that artificial dataset. When this procedure is
repeated often enough, the mean values and standard deviations of fit 
parameters are calculated.
