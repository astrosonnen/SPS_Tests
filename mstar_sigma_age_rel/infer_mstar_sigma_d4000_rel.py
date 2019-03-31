import numpy as np
import emcee
from scipy.stats import truncnorm
import h5py


f = open('../lega-c_sersic_it2_stellarpop_pars.cat', 'r')
mstar, merr, t90, reff, sigma, sigma_err, hb_ew, oii_ew, d4000, hd_ew = np.loadtxt(f, usecols=(6, 7, 10, 13, 16, 17, 19, 20, 21, 22), unpack=True)
f.close()

good = (hb_ew == hb_ew) & (sigma_err > 0.) & (sigma > 0.) & (sigma < 600.) & (d4000 == d4000) & (d4000 > 0.)
quiesc = hb_ew > -3.

keep = good & quiesc
ngal = keep.sum()

lmstar = mstar[keep]
lmstar_err = merr[keep]
lsigma = np.log10(sigma[keep])
lsigma_err = 0.5*(np.log10(sigma + sigma_err) - np.log10(sigma - sigma_err))[keep]
d4000 = d4000[keep]

lmstar_impsamp = []
for i in range(ngal):
    lmstar_impsamp.append(np.random.normal(lmstar[i], lmstar_err[i], 1000))

mstar_piv = 11.

nstep = 500

mstar_mu = {'name': 'mstar_mu', 'lower': 10., 'upper': 13., 'guess': lmstar.mean(), 'step': 0.03}
mstar_sig = {'name': 'mstar_sig', 'lower': 0., 'upper': 1., 'guess': lmstar.std(), 'step': 0.03}

sigma_mu = {'name': 'sigma_mu', 'lower': 1., 'upper': 3., 'guess': 2.2, 'step': 0.1}
sigma_sig = {'name': 'sigma_sig', 'lower': 0., 'upper': 1., 'guess': 0.05, 'step': 0.03}
sigma_mstar_dep = {'name': 'sigma_mstar_dep', 'lower': -3., 'upper': 3., 'guess': 0.3, 'step': 0.03}

d4000_mu = {'name': 'd4000_mu', 'lower': 0., 'upper': 5., 'guess': d4000.mean(), 'step': 0.1}
d4000_sig = {'name': 'd4000_sig', 'lower': 0., 'upper': 3., 'guess': d4000.std(), 'step': 0.03}
d4000_mstar_dep = {'name': 'd4000_mstar_dep', 'lower': -3., 'upper': 3., 'guess': 0., 'step': 0.03}
d4000_sigma_dep = {'name': 'd4000_sigma_dep', 'lower': -3., 'upper': 3., 'guess': 0., 'step': 0.03}

pars = [mstar_mu, mstar_sig, sigma_mu, sigma_sig, sigma_mstar_dep, d4000_mu, d4000_sig, d4000_mstar_dep, d4000_sigma_dep]

npars = len(pars)

nwalkers = 6*npars

bounds = []
for par in pars:
    bounds.append((par['lower'], par['upper']))

start = []
for i in range(nwalkers):
    tmp = np.zeros(npars)
    for j in range(npars):
        a, b = (bounds[j][0] - pars[j]['guess'])/pars[j]['step'], (bounds[j][1] - pars[j]['guess'])/pars[j]['step']
        p0 = truncnorm.rvs(a, b, size=1)*pars[j]['step'] + pars[j]['guess']
        tmp[j] = p0

    start.append(tmp)

def logprior(p):
    for i in range(npars):
        if p[i] < bounds[i][0] or p[i] > bounds[i][1]:
            return -1e300
    return 0.

def logpfunc(p):

    lprior = logprior(p)
    if lprior < 0.:
        return -1e300

    mstar_mu, mstar_sig, sigma_mu, sigma_sig, sigma_mstar_dep, d4000_mu, d4000_sig, d4000_mstar_dep, d4000_sigma_dep = p

    logp = 0.
    for i in range(ngal):
        d4000_muhere = d4000_mu + d4000_mstar_dep * (lmstar_impsamp[i] - mstar_piv)

        sigma_muhere = sigma_mu + sigma_mstar_dep * (lmstar_impsamp[i] - mstar_piv)
        sigma_mueff = sigma_muhere + (d4000[i] - d4000_muhere)/d4000_sigma_dep

        sigma_sigeff = d4000_sig/abs(d4000_sigma_dep)

        mu1 = lsigma[i]
        mu2 = sigma_muhere
        mu3 = sigma_mueff

        sigma1 = lsigma_err[i]
        sigma2 = sigma_sig
        sigma3 = sigma_sigeff

        sigma_eff = (1./sigma1**2 + 1./sigma2**2 + 1./sigma3**2)**(-0.5)
        mu_eff = sigma_eff**2 * (mu1/sigma1**2 + mu2/sigma2**2 + mu3/sigma3**2)

        sigma_integral = 1./(2.*np.pi)*sigma_eff/sigma1/sigma2/sigma3/abs(d4000_sigma_dep) * np.exp(-0.5*(mu1**2/sigma1**2 + mu2**2/sigma2**2 + mu3**2/sigma3**2 - mu_eff**2/sigma_eff**2))

        mstar_term = 1./(2.*np.pi)**0.5/mstar_sig * np.exp(-0.5*(lmstar_impsamp[i] - mstar_mu)**2/mstar_sig**2)

        integrand = mstar_term * sigma_integral

        logp += np.log(integrand.sum())

    if logp != logp:
        return -1e300

    return logp

sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc, threads=50)

print "Sampling on %d galaxies"%ngal

sampler.run_mcmc(start, nstep)

ml = sampler.lnprobability.argmax()
for n in range(npars):
    print '%s %4.3f'%(pars[n]['name'], sampler.chain[:, :, n].flatten()[ml])

output = h5py.File('mstar_sigma_d4000_rel.hdf5', 'w')
output.create_dataset('logp', data=sampler.lnprobability)
for n in range(npars):
    output.create_dataset(pars[n]['name'], data=sampler.chain[:, :, n])

