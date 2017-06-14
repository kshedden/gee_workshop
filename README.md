Generalized Estimating Equations (GEE) in Python Statsmodels
============================================================

https://github.com/statsmodels/statsmodels/wiki/Examples

https://github.com/statsmodels/statsmodels/blob/master/statsmodels/genmod/generalized_estimating_equations.py

https://github.com/statsmodels/statsmodels/blob/master/statsmodels/genmod/generalized_linear_model.py

https://github.com/statsmodels/statsmodels/blob/master/statsmodels/genmod/cov_struct.py

https://support.sas.com/documentation/cdl/en/statug/63347/HTML/default/viewer.htm#statug_genmod_sect049.htm

http://www.stata.com/manuals13/xtxtgee.pdf

__Generalized Linear Models__

Generalized linear models (GLMs) are a family of regression
procedures.  Like most basic forms of regression, they relate a
dependent variable y to one or more independent variables (also called
predictors or covariates) x1, ..., xp.  The conditional mean E[y|x1,
..., xp] and conditional variance Var[y|x1, ..., xp] play important
roles in GLMs.

GLMs have the following properties:

* They are _single index models_, meaning that the fitted mean value
  for an observation with covariate values x1, x2, ..., xp depends on
  the covariates only through a linear function b0 + b1×x1 + ... +
  bp×xp, where the bj are unknown parameters.

* They have a mean/variance relationship, i.e. Var[y|x1, ..., xp] is a
  function of E[y|x1, ..., xp].

* They may specify a limited domain for the dependent variable,
  e.g. the interval [0, 1], or the non-negative integers.

* The distribution of y given x1, ..., xp follows an exponential
  family.

We won't cover the theory of GLMs here in any detail.  The main thing
to appreciate about GLMs is that a specific GLM is specified by
choosing a _family_ and a _link function_.  The link function is the
function that maps the mean E[y|x1, ..., xp] to the linear predictor.

The family and link function together imply the mean/variance
relationship and the domain of the response variable.  Most families
can be used with several different link functions.  One link function
for each family is _canonical_, meaning that it has special properties
that simplify working with the model.

Below are some of the common GLM families and link functions:

* _Gaussian_: a Gaussian GLM is equivalent to linear least squares
  regression.  Although it is interesting that linear least squares
  can be viewed in this framework, Gaussian GLMs are less commonly
  used than OLS regression.  The canonical link function for the
  Gaussian family is the identify function g(x) = x.  The domain for
  the Gaussian GLM is the real line (i.e. there is no constraint on
  the dependent variable's values).

* _Binomial_: a binomial GLM is also known as "logistic regression".
  This is probably the most common GLM.  In this setting, the response
  variable takes on only two distinct values, usually coded 0 and 1.
  The canonical link function for the binomial GLM is the logit
  function log(p/(1-p)), where p = E[y|x1, ..., xp] is the mean.
  Other link functions used with the binomial GLM are the log function
  (giving a "log binomial model") and the inverse cumulative
  distribution function of the Gaussian distribution (giving "probit
  regression").  The domain for the binomial GLM is the set of two
  values that y can take on (e.g. 0, 1).

* _Poisson_: in a Poisson GLM, the distribution of y given x1, ..., xp
  is Poisson, with mean exp(b0 + b1×x1 + ... + bp×xp) in the canonical
  link case.  A Poisson distribution has the property that the mean is
  equal to the variance, so the Poisson GLM is useful for modeling
  data with this relationship.  The domain for the Poisson GLM is the
  non-negative integers 0, 1, ...  The canonical link for the Poisson
  GLM is the log function.

* _Negative binomial_: this GLM generalizes the Poisson GLM by adding
  an additional parameter alpha to the mean/variance relationship.
  The variance of y is equal to m + alpha×m^2, where m is the
  conditional mean.  If alpha = 0, this is the same as the Poisson
  mean/variance relationship.  By setting alpha > 0, the variance can
  grow faster than the mean.  The canonical link for the negative
  binomial GLM is the reciprocal function, but in practice people
  usually use the non-canonical log link.

Other GLM families that we will not discuss here are the Gamma GLM,
the inverse Gaussian GLM, and the Tweedie GLM.

__Generalized Estimating Equations (GEE)__

GLMs are useful in many settings where the observations are
independent.  In practice, we often encounter dependent data.  Some
examples where dependent data arise are: longitudinal data, other
forms of repeated measures on subjects, clustered data such as data
observed on multiple subjects in a cluster (e.g. classroom, hospital),
or data collected by geographic region (e.g. census tracts).

If the main interest is in the mean structure, it is usually
meaningful to use GLMs even if the data are dependent.  However the
inferential parts of the analysis (standard errors, confidence
intervals, p-values, etc.) will usually be incorrect if the data are
dependent and are analyzed using a GLM.  The _marginal mean structure_
is estimated correctly with GLM models even in the presence of
dependence.  This marginal mean structure has a related but different
interpretation than the conditional mean structure that is estimated
using mutilevel models.  This is an important distinction but we will
not explore it in detail here.

GEE was developed to allow GLM-style analysis to be performed on
dependent data.  It is different from many other statistical
techniques that involve models.  Instead of a model, GEE is based on a
set of estimating equations.  These estimating equations involve the
GLM mean structure, and a _working covariance structure_ (that need
not be correct, more about this later).  Solving these equations
yields estimates of the marginal mean structure parameters (regression
coefficients), and provides a means to obtain standard errors that
properly account for the dependence in the data.

The working covariance structure plays an important role in GEE
analysis.  Most GEE covariance models are based on the notion of the
data being dependent within clusters.  Other dependence structures
such as nested clusters and sequential structures are also possible.
A wide variety of working covariance structures can be specified.
Here are some of the more common ones:

* _Independence_: this working covariance structure treats the
  observations as being independent.

* _Exchangeable_: this working covariance structure treats any two
  observations within a cluster has having a constant, unknown
  correlation parameter r.  Pairs of observations in different
  clusters are taken to be independent.

* _Autoregressive_: this working covariance treats the observations
  within a cluster as having correlation that depends exponentially on
  the distance between the observations, i.e. the correlation between
  consecutive observations is r, that between observations separated
  by one value is r^2, etc.

Other covariance structures include m-dependent covariance for
stationary time series, unstructured covariances for vector data, and
various specialized covariance structures for ordinal or categorical
data.

GEE inference can be used in two ways.  Taking the working covariance
structure to be true, we have a form of parametric analysis.  This is
sometimes called "naive inference" for GEE.  When using naive
inference, it is important to assess whether the specified covariance
is flexible enough to fit the data.  Alternatively, we take the
covariance structure to be a "working structure" that does not need to
be true.  This gives rise to a form of "robust inference" that
accommodates covariance mis-specification.

There are a number of subtleties that arise when using GEE.  In a
fully model-based analysis, the main focus is on the structure and fit
of the model.  In a GEE, there are other choices to be made that
impact the performance of the mean structure estimation.  Here are a
few of the important issues to consider:

* The robust approach to GEE works quite well for large data sets, but
  the covariance estimates are quite variable for smaller samples.

* The mean and covariance estimates are generally not orthogonal --
  changing the covariance structure can lead to changes in the
  estimated mean structure.  If this is undesirable, use the
  independence covariance structure, in which case the mean parameters
  will be the same as when analyzing the data with GLM.

* Most covariance structures have parameters that are estimated using
  the method of moments applied to the residuals from the mean
  estimation.  The mean structure parameters and covariance structure
  parameters are estimated in separate steps.  It is common to
  alternate between them to try to achieve convergence (although
  convergence does not always occur).  Unlike in MLE and other
  iterative optimizations, it is not necessary to iterate to
  convergence -- the iterations can stop at any point, including after
  the first covariance update, and meaningful estimates will generally
  result.

* Since the analysis is not based on likelihoods, likelihood-based
  procedures such as the likelihood ratio test (LRT), or model
  selection procedures including AIC and BIC cannot be directly
  applied.  Wald tests and a type of score test can be applied, and
  modified versions of the AIC are available.

* Covariance parameter estimates (for correctly-specified models) are
  generally consistent, but there is no straightforward way to assess
  uncertainty for these estimates.  GEE is therefore mainly used when
  the primary questions are about the mean structure.

* As noted above, GEE can be used to estimate the marginal regression
  function.  In a linear model, the marginal mean structure and
  conditional mean structure are the same (although the estimates of
  mean structure parameters from GEE and mixed modeling will not
  coincide).  For nonlinear models (e.g. logistic GLM), the marginal
  mean structure and conditional mean structures differ.  In general,
  the marginal effects are smaller than the conditional effects
  (e.g. when expressed as odds ratios).