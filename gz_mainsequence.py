# Branch - make all plots using only a lower limit of M_star = 10**8 M_sun

import numpy as np
import time

from matplotlib import rc
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.font_manager import FontProperties
from astropy.io import fits
from astropy import cosmology
from astropy import units as u

from scipy.stats import ks_2samp
from scipy.linalg.basic import LinAlgError
from astroML.plotting import hist as histML

import warnings
warnings.simplefilter(action = "ignore", category = (FutureWarning,np.RankWarning))

#plt.ion()

gz_path = '/Users/willettk/Astronomy/Research/GalaxyZoo'
gzm_path = '%s/gzmainsequence' % gz_path
fig_path = '%s/paper/figures/masslim' % gzm_path
fits_path = '%s/fits' % gz_path

aw_tasks = ['t10_arms_winding_'+r for r in ('a28_tight','a29_medium','a30_loose')]
an_tasks = ['t11_arms_number_'+r for r in ('a31_1','a32_2','a33_3','a34_4','a36_more_than_4','a37_cant_tell')]
an_colors = ('red','orange','yellow','green','blue','purple')
aw_colors = ('red','green','blue')

z_lowerlim = 0.000      # any reason for a lower limit?
z_upperlim = 0.085      # upper limit of reliable GZ2 classification
absmag_lim = -19.5
mass_lim = 1e8

def get_data():

    filename = '%s/mpajhu_gz2.fits' % fits_path

    with fits.open(filename) as f:
        data = f[1].data

    return data

def get_sample(data,starforming=True):

    # Find starforming galaxies

    sf = (data['bpt'] == 1) #| (data['bpt'] == 2)
    redshift = (data['REDSHIFT'] <= z_upperlim) & (data['REDSHIFT'] >= z_lowerlim)
    #absmag = data['PETROMAG_MR'] < absmag_lim        # could use 0.085 and -20.17, or 0.05 and -19.5

    # Restrict sample to galaxies above 10**8 M_sun (completeness limit of MPA-JHU; Brinchmann+04)
    masslim = (data['MEDIAN_MASS'] >= 8.)

    if starforming:
        #sample = data[sf & redshift & absmag]
        sample = data[sf & redshift & masslim]
    else:
        #sample = data[redshift & absmag]
        sample = data[redshift]


    return sample

def get_mass_sfr(sf_sample):

    mass_sfr_good = (sf_sample['MEDIAN_MASS'] > -99) & (sf_sample['MEDIAN_SFR'] > -99)

    mass = sf_sample[mass_sfr_good]['MEDIAN_MASS']
    sfr = sf_sample[mass_sfr_good]['MEDIAN_SFR']
    sfr16_err = np.abs(sf_sample[mass_sfr_good]['P16_SFR'] - sf_sample[mass_sfr_good]['MEDIAN_SFR'])
    sfr84_err = np.abs(sf_sample[mass_sfr_good]['P84_SFR'] - sf_sample[mass_sfr_good]['MEDIAN_SFR']) 
    sfr_err = (sfr16_err + sfr84_err) / 2.

    # Specific star formation rate is SFR/mass; in log terms, simple subtraction
    ssfr = sfr - mass
    ssfr_err = sfr_err - mass

    return mass,sfr,sfr_err,ssfr,ssfr_err

def bins(mass_min=6,mass_max=13,sfr_min=-5,sfr_max=2,ssfr_min=-12,ssfr_max=-8,nbins=50):

    mass_bins = np.linspace(mass_min,mass_max,nbins)
    sfr_bins = np.linspace(sfr_min,sfr_max,nbins)
    ssfr_bins = np.linspace(ssfr_min,ssfr_max,nbins)

    return mass_bins,sfr_bins,ssfr_bins
    
def plot_fits(label,mass,yval,yval_weights,axis,color,morph='t11_arms_number_a31_1',lw=2,ls='--',legend=False,verbose=False):

    prefix = 'un' if yval_weights != None else ''

    (a,b),v = np.polyfit(mass,yval,1,full=False,w=yval_weights,cov=True)

    xarr = np.linspace(6,12,100)
    line = axis.plot(xarr,np.polyval([a,b],xarr),color=color,linestyle=ls,linewidth=lw)

    if verbose:
        print 'Best fit (%10s) for %sweighted data: a,b = %.2f,%.2f' % (label,prefix,a,b)
        print 'Best fit (%10s) for %sweighted data: sigma_a,sigma_b=%.2e,%.2e' % (label,prefix,v[0,0],v[1,1])
        print ''

    return a,b

def plot_ms_arms_number(sf_sample,weighted=False,contour=False,plot_ssfr=False,verbose=False):

    # Plot

    fig = plt.figure(1,(10,6))
    fig.clf()
    filestr=''

    fig.subplots_adjust(left=0.08,bottom=0.15,right=0.9,hspace=0,wspace=0)

    mass,sfr,sfr_err,ssfr,ssfr_err = get_mass_sfr(sf_sample)
    mass_bins,sfr_bins,ssfr_bins = bins()

    if plot_ssfr:
        yval = ssfr
        yval_err = ssfr_err
        yval_bins = ssfr_bins
    else:
        yval = sfr
        yval_err = sfr_err
        yval_bins = sfr_bins

    if weighted:
        filestr += '_weighted'
    if contour:
        filestr += '_contour'
    if plot_ssfr:
        filestr += '_ssfr'

    h,xedges,yedges = np.histogram2d(mass,yval,bins=(mass_bins,yval_bins))

    # Plot each one

    responses = ('a31_1','a32_2','a33_3','a34_4','a36_more_than_4','a37_cant_tell')
    arm_tasks = tuple(['t11_arms_number_'+r for r in responses])
    arm_label = ('1','2','3','4','5+','??')
    colors = ('red','orange','yellow','green','blue','purple')
    for idx, (a,c,label) in enumerate(zip(arm_tasks,colors,arm_label)):

        spiral = (sf_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sf_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sf_sample['t04_spiral_a08_spiral_weight'] >= 20) & (sf_sample['t04_spiral_a08_spiral_debiased'] > 0.619)
        spirals = sf_sample[spiral]
        morph_sample = sf_sample[spiral & (sf_sample['%s_flag' % a] == 1)] 

        ax = fig.add_subplot(2,3,idx+1)
        h = ax.hist2d(mass,yval,bins=50,cmap = cm.gray_r, norm=LogNorm())
        

        if plot_ssfr:
            ax.set_ylim(-12,-8)
            ylabel='sSFR '+r'$[\log\/\mathrm{yr}^{-1}]$'
            yval_weighted = spirals['MEDIAN_SFR'] - spirals['MEDIAN_MASS']
            yval_scatter = morph_sample['MEDIAN_SFR'] - morph_sample['MEDIAN_MASS']
            ytextloc = -11.5
        else:
            ax.set_ylim(-2.9,2)
            ylabel='SFR '+r'$[\log\/M_\odot/\mathrm{yr}]$'
            yval_weighted = spirals['MEDIAN_SFR']
            yval_scatter = morph_sample['MEDIAN_SFR']
            ytextloc = 1.0

        if idx == 4:
            ax.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',fontsize=16)
        if idx < 3:
            ax.get_xaxis().set_ticks([])
        else:
            ax.get_xaxis().set_ticks([8.,9.,10.,11.])

        if idx == 0 or idx == 3:
            ax.set_ylabel(ylabel,fontsize=16)
        else:
            ax.get_yaxis().set_ticks([])


        # Two sets of plots: one weights histogram by debiased vote fraction per galaxy; other shows discrete categories from GZ2 flags.
        if weighted:
            if contour:
                hc,xc,yc = np.histogram2d(spirals['MEDIAN_MASS'],yval_weighted,bins=(mass_bins,yval_bins),weights=spirals['%s_debiased' % a])
                levels=10**(np.linspace(0,2,8))
                CS = ax.contour(mass_bins[1:],yval_bins[1:],hc.T,levels,colors=c)
            else:
                h = ax.hist2d(spirals['MEDIAN_MASS'],yval_weighted,bins=50,cmap = cm.RdYlGn, weights=spirals['%s_debiased' % a],vmin=0.01,vmax=100.,norm=LogNorm())

            cb_label = r'$w_\mathrm{\phi}$'

            # Weight by debiased vote fraction
            yval_weights = spirals['%s_debiased' % a]

            plot_fits(label,spirals['MEDIAN_MASS'],yval_weighted,yval_weights,ax,c,morph=a,verbose=verbose)
        else:
            ax.scatter(morph_sample['MEDIAN_MASS'],yval_scatter, s = 2, color=c, marker='o')
            cb_label = r'$N_\mathrm{star-forming\/galaxies}$'

            # No weighting

            plot_fits(label,morph_sample['MEDIAN_MASS'],yval_scatter,None,ax,c,verbose=verbose)

        ax.text(8.2,ytextloc,r'$N_{arms} = $%s' % label, color='k',fontsize=18)

        # Plot the linear fits for all star-forming galaxies
        plot_fits('SF galaxies',mass,yval,None,ax,'black',lw=1,ls='-',verbose=verbose)
        ax.set_xlim(8,11.5)

        '''
        # Print number of galaxies in each category

        print '%i %s galaxies' % (len(morph_sample),a)
        '''


    box = ax.get_position()
    axColorbar = plt.axes([box.x0*1.05 + box.width * 0.95, box.y0, 0.01, box.height*2])
    cb = plt.colorbar(h[3],cax = axColorbar, orientation="vertical")
    cb.set_label(r'$N_\mathrm{star-forming\/galaxies}$' ,fontsize=16)

    fig.savefig('%s/ms_arms_number%s.pdf' % (fig_path,filestr), dpi=200)

    return None

def plot_ms_arms_winding(sf_sample,weighted=False,contour=False,plot_ssfr=False,verbose=False):

    # Plot

    fig = plt.figure(2,(10,4))
    fig.clf()
    filestr=''

    fig.subplots_adjust(left=0.08,bottom=0.15,hspace=0,wspace=0)

    mass,sfr,sfr_err,ssfr,ssfr_err = get_mass_sfr(sf_sample)
    mass_bins,sfr_bins,ssfr_bins = bins()

    if plot_ssfr:
        yval = ssfr
        yval_err = ssfr_err
        yval_bins = ssfr_bins
    else:
        yval = sfr
        yval_err = sfr_err
        yval_bins = sfr_bins

    if weighted:
        filestr += '_weighted'
    if contour:
        filestr += '_contour'
    if plot_ssfr:
        filestr += '_ssfr'

    h,xedges,yedges = np.histogram2d(mass,sfr,bins=(mass_bins,yval_bins))

    # Plot each morphological category

    responses = ('a28_tight','a29_medium','a30_loose')
    arm_tasks = tuple(['t10_arms_winding_'+r for r in responses])
    colors = ('red','green','blue')
    for idx, (a,c) in enumerate(zip(arm_tasks,colors)):

        arm_label = a[21:]
        spiral = (sf_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sf_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sf_sample['t04_spiral_a08_spiral_weight'] >= 20) & (sf_sample['t04_spiral_a08_spiral_debiased'] > 0.619)
        spirals = sf_sample[spiral]
        morph_sample = sf_sample[spiral & (sf_sample['%s_flag' % a] == 1)] 

        ax = fig.add_subplot(1,3,idx+1)
        h = ax.hist2d(mass,yval,bins=50,cmap = cm.gray_r, norm=LogNorm())

        if plot_ssfr:
            ax.set_ylim(-12,-8)
            ytextloc = -11.5
            ylabel='sSFR '+r'$[\log\/\mathrm{yr}^{-1}]$'
            yval_weighted = spirals['MEDIAN_SFR'] - spirals['MEDIAN_MASS']
            yval_scatter = morph_sample['MEDIAN_SFR'] - morph_sample['MEDIAN_MASS']
        else:
            ax.set_ylim(-3,2)
            ytextloc = 1.3
            ylabel='SFR '+r'$[\log\/M_\odot/\mathrm{yr}]$'
            yval_weighted = spirals['MEDIAN_SFR']
            yval_scatter = morph_sample['MEDIAN_SFR']

        if idx == 0:
            ax.set_ylabel(ylabel,fontsize=16)
        if idx == 1:
            ax.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',fontsize=16)

        if idx > 0:
            ax.get_yaxis().set_ticks([])

        ax.get_xaxis().set_ticks([8.,9.,10.,11.])

        # Two sets of plots: one weights histogram by debiased vote fraction per galaxy; other shows discrete categories from GZ2 flags.
        if weighted:
            if contour:
                hc,xc,yc = np.histogram2d(spirals['MEDIAN_MASS'],yval_weighted,bins=(mass_bins,yval_bins),weights=spirals['%s_debiased' % a])
                levels=10**(np.linspace(0,2,8))
                CS = ax.contour(mass_bins[1:],yval_bins[1:],hc.T,levels,colors=c)
                cb_label = r'$N_\mathrm{star-forming\/galaxies}$'
            else:
                h = ax.hist2d(spirals['MEDIAN_MASS'],yval_weighted,bins=50,cmap = cm.RdYlGn, weights=spirals['%s_debiased' % a],vmin=0.01,vmax=100.,norm=LogNorm())
                cb_label = r'$w_\mathrm{\phi}$'

            #spirals = sf_sample[spiral]
            #h = ax.hist2d(spirals['MEDIAN_MASS'],yval_weighted,bins=50,cmap = cm.RdYlGn, weights=spirals['%s_debiased' % a],vmin=0.01,vmax=100.,norm=LogNorm())
            #cb_label = r'$w_\mathrm{\phi}$'

            yval_weights = spirals['%s_debiased' % a]

            plot_fits(arm_label,spirals['MEDIAN_MASS'],yval_weighted,yval_weights,ax,c,morph=a,verbose=verbose)
        else:
            ax.scatter(morph_sample['MEDIAN_MASS'],yval_scatter, s = 2, color=c, marker='o')
            cb_label = r'$N_\mathrm{star-forming\/galaxies}$'

            # No weighting
            yval_weights = None
            plot_fits(arm_label,morph_sample['MEDIAN_MASS'],yval_scatter,yval_weights,ax,c,morph=a,verbose=verbose)

        if plot_ssfr:
            ax.set_ylim(-12,-8)
        else:
            ax.set_ylim(-3,2)

        ax.text(8.2,ytextloc,r'$\phi_{arms} = $%s' % arm_label, color='k')

        # Plot the best linear fits

        plot_fits('SF galaxies',mass,yval,None,ax,'black',lw=1,ls='-')
        ax.set_xlim(8,11.5)

        '''
        # Print number of galaxies in each category

        print '%i %s galaxies' % (len(morph_sample),a)
        '''

    # Set the colorbar and labels at the end

    box = ax.get_position()
    axColorbar = plt.axes([box.x0*1.05 + box.width * 0.93, box.y0, 0.01, box.height])
    cb = plt.colorbar(h[3],cax = axColorbar, orientation="vertical")
    cb.set_label(cb_label,fontsize=16)

    fig.savefig('%s/ms_arms_winding%s.pdf' % (fig_path,filestr), dpi=200)

    return None

def plot_ms_bars(sf_sample,contour=False,plot_ssfr=False,verbose=False):

    # Plot

    fig = plt.figure(3,(11,6))
    fig.clf()
    filestr=''

    fig.subplots_adjust(left=0.08,bottom=0.15,right=0.9,hspace=0,wspace=0)

    mass,sfr,sfr_err,ssfr,ssfr_err = get_mass_sfr(sf_sample)
    mass_bins,sfr_bins,ssfr_bins = bins()

    if plot_ssfr:
        yval = ssfr
        yval_err = ssfr_err
        yval_bins = ssfr_bins
    else:
        yval = sfr
        yval_err = sfr_err
        yval_bins = sfr_bins

    h,xedges,yedges = np.histogram2d(mass,sfr,bins=(mass_bins,yval_bins))

    if contour:
        filestr += '_contour'
    if plot_ssfr:
        filestr += '_ssfr'

    notedgeon = (sf_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sf_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sf_sample['t02_edgeon_a05_no_weight'] >= 20)

    barred = sf_sample[notedgeon & (sf_sample['t03_bar_a06_bar_debiased'] >= 0.4)] 
    unbarred = sf_sample[notedgeon & (sf_sample['t03_bar_a06_bar_debiased'] < 0.4)] 

    bar_data = (barred,unbarred)
    bar_text = ('Barred','Unbarred')
    color = ('blue','red')

    # Plot barred and unbarred
    for idx, (b,c,t) in enumerate(zip(bar_data,color,bar_text)):

        ax = fig.add_subplot(1,2,idx+1)
        h2 = ax.hist2d(mass,yval,bins=50,cmap = cm.gray_r, norm=LogNorm())
        ax.set_ylim(-3,2)
        ax.set_xlabel('Stellar mass (log '+r'$M/M_\odot$)',fontsize=20)

        if plot_ssfr:
            ax.set_ylim(-12,-8)
            ytextloc = -11.5
            ylabel='sSFR '+r'$[\log\/\mathrm{yr}^{-1}]$'
            yval_morph = b['MEDIAN_SFR'] - b['MEDIAN_MASS']
        else:
            ax.set_ylim(-3,2)
            ytextloc = 1.4
            ylabel='SFR '+r'$[\log\/M_\odot/\mathrm{yr}]$'
            yval_morph = b['MEDIAN_SFR']

        if idx == 0:
            ax.set_ylabel('SFR '+r'$[\log\/M_\odot/\mathrm{yr}]$',fontsize=20)
            ax.get_xaxis().set_ticks([8.0,8.5,9.0,9.5,10.0,10.5,11.0])
        else:
            ax.get_yaxis().set_ticks([])

        # Contour plot

        if contour:
            hb,xb,yb = np.histogram2d(b['MEDIAN_MASS'],yval_morph,bins=(mass_bins,yval_bins))
            levels=10**(np.linspace(0,2,8))
            CS = ax.contour(mass_bins[1:],yval_bins[1:],hb.T,levels,colors=c)
        else:
            ax.scatter(b['MEDIAN_MASS'],yval_morph, s=2, color=c, marker='o')

        ax.text(8.2,ytextloc,t,color=c,fontsize=18)

        # Plot the best linear fit

        plot_fits(t,b['MEDIAN_MASS'],yval_morph,None,ax,c,verbose=verbose)
        plot_fits(t,mass,yval,None,ax,'black',lw=1,ls='-')
        ax.set_xlim(8,11.5)

        '''
        # Print counts

        print '%6i %s galaxies' % (len(b),t)
        '''


    box = ax.get_position()
    axColorbar = plt.axes([box.x0*1.05 + box.width * 0.95, box.y0, 0.01, box.height])
    cb = plt.colorbar(h2[3],cax = axColorbar, orientation="vertical")
    cb.set_label(r'$N_\mathrm{star-forming\/galaxies}$' ,fontsize=18)

    fig.savefig('%s/ms_bar%s.pdf' % (fig_path,filestr), dpi=200)

    return None
    
def plot_ms_greenpeas(sf_sample):

    # Plot

    fig = plt.figure(5,(10,8))
    fig.clf()

    fig.subplots_adjust(left=0.08,bottom=0.15,hspace=0,wspace=0)

    mass,sfr,sfr_err,ssfr,ssfr_err = get_mass_sfr(sf_sample)
    mass_bins,sfr_bins,ssfr_bins = bins()
    h,xedges,yedges = np.histogram2d(mass,sfr,bins=(mass_bins,sfr_bins))

    # Plot star-forming galaxies

    ax = fig.add_subplot(111)
    h = ax.hist2d(mass,sfr,bins=50,cmap = cm.gray_r, norm=LogNorm())
    ax.set_xlim(8,11.5)
    ax.set_ylim(-3,2)
    ax.set_ylabel('log SFR '+r'$[M_\odot/\mathrm{yr}]$',fontsize=16)
    ax.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',fontsize=16)

    # Plot green peas

    with fits.open('%s/greenpeas.fits' % gzm_path) as f:
        data = f[1].data

    sc = ax.scatter(data['M_STELLAR'],np.log10(data['SFR']), color='green',s = 10, marker='o')

    ax.text(8.2,1.3,'Green peas', color='green')

    # Plot the best linear fits

    a1,a0 = plot_fits('Star-forming galaxies',mass,sfr,None,ax,'black',lw=1,ls='-')

    # Set the colorbars and labels

    box = ax.get_position()
    axColorbar = plt.axes([box.x0 + box.width * 1.02, box.y0, 0.01, box.height])
    cb = plt.colorbar(h[3],cax = axColorbar, orientation="vertical")
    cb.set_label(r'$N_\mathrm{star-forming\/galaxies}$' ,fontsize=16)

    fig.savefig('%s/ms_greenpeas.pdf' % fig_path, dpi=200)

    return None

def sigma_mstar(sf_sample):

    '''
    Test whether populations are likely to be consistent by plotting dispersion as function of M_star for bins of M_star
    Overplot for different populations
    '''

    mass,sfr,sfr_err,ssfr,ssfr_err = get_mass_sfr(sf_sample)
    mass_bins,sfr_bins,ssfr_bins = bins(mass_max=11.5,nbins=15)
    deltamass = mass_bins[1] - mass_bins[0]

    fig = plt.figure(6,(10,8))
    fig.clf()

    # Bars
    ax = fig.add_subplot(223)

    sigma_sfr = []
    sigma_sfr_err = []
    for mb in mass_bins:
        idx = (mass > (mb-deltamass)) & (mass <= (mb+deltamass))
        sigma_sfr.append(np.std(sfr[idx]))
        try:
            sigma_sfr_err.append(1./np.sum(idx))
        except ZeroDivisionError:
            sigma_sfr_err.append(0.)

    ax.errorbar(mass_bins,sigma_sfr,yerr=sigma_sfr_err,fmt='o',color='black',capsize=0)

    notedgeon = (sf_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sf_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sf_sample['t02_edgeon_a05_no_weight'] >= 20)
    barred = sf_sample[notedgeon & (sf_sample['t03_bar_a06_bar_debiased'] >= 0.4)] 
    unbarred = sf_sample[notedgeon & (sf_sample['t03_bar_a06_bar_debiased'] < 0.4)] 

    bar_data = (barred,unbarred)
    bar_text = ('Barred','Unbarred')
    color = ('blue','red')

    for idx, (b,c,t) in enumerate(zip(bar_data,color,bar_text)):

        sigma_sfr,sigma_sfr_err,ngals = [],[],[]
        for mb in mass_bins:
            idx = (b['MEDIAN_MASS'] > (mb-deltamass)) & (b['MEDIAN_MASS'] <= (mb+deltamass))
            sigma_sfr.append(np.std(b['MEDIAN_SFR'][idx]))
            try:
                sigma_sfr_err.append(1./np.sum(idx))
            except ZeroDivisionError:
                sigma_sfr_err.append(0.)
            ngals.append(np.sum(idx))

        gtr10 = (np.array(ngals) >= 10)
        ax.errorbar(mass_bins[gtr10],np.array(sigma_sfr)[gtr10],yerr=np.array(sigma_sfr_err)[gtr10],color=c,capsize=0)

    # Final plot labels and ranges
    ax.set_xlim(8,11.5)
    ax.set_ylim(0,1.2)
    ax.set_ylabel(r'$\sigma_{SFR}~[\log M_\odot/\mathrm{yr}]$',fontsize=16)
    ax.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',fontsize=16)
    ax.set_title('Bar/no bar')

    # Arm number
    ax = fig.add_subplot(221)

    sigma_sfr = []
    sigma_sfr_err = []
    for mb in mass_bins:
        idx = (mass > (mb-deltamass)) & (mass <= (mb+deltamass))
        sigma_sfr.append(np.std(sfr[idx]))
        try:
            sigma_sfr_err.append(1./np.sum(idx))
        except ZeroDivisionError:
            sigma_sfr_err.append(0.)

    ax.errorbar(mass_bins,sigma_sfr,yerr=sigma_sfr_err,fmt='o',color='black',capsize=0)

    arm_tasks = ('a31_1','a32_2','a33_3','a34_4','a36_more_than_4','a37_cant_tell')
    arm_label = ('1','2','3','4','5+','??')
    colors = ('red','orange','yellow','green','blue','purple')
    for idx, (a,c,al) in enumerate(zip(arm_tasks,colors,arm_label)):

        spiral = (sf_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sf_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sf_sample['t04_spiral_a08_spiral_weight'] >= 20) & (sf_sample['t04_spiral_a08_spiral_debiased'] > 0.619)
        morph_sample = sf_sample[spiral & (sf_sample['t11_arms_number_%s_flag' % a] == 1)] 

        sigma_sfr,sigma_sfr_err,ngals = [],[],[]
        for mb in mass_bins:
            idx = (morph_sample['MEDIAN_MASS'] > (mb-deltamass)) & (morph_sample['MEDIAN_MASS'] <= (mb+deltamass))
            sigma_sfr.append(np.std(morph_sample['MEDIAN_SFR'][idx]))
            try:
                sigma_sfr_err.append(1./np.sum(idx))
            except ZeroDivisionError:
                sigma_sfr_err.append(0.)
            ngals.append(np.sum(idx))

        gtr10 = (np.array(ngals) >= 10)
        ax.errorbar(mass_bins[gtr10],np.array(sigma_sfr)[gtr10],yerr=np.zeros_like(sigma_sfr)[gtr10],color=c,capsize=0)

    # Final plot labels and ranges
    ax.set_xlim(8,11.5)
    ax.set_ylim(0,1.2)
    ax.set_ylabel(r'$\sigma_{SFR}~[\log M_\odot/\mathrm{yr}]$',fontsize=16)
    ax.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',fontsize=16)
    ax.set_title('Arms number')

    # Arm winding
    ax = fig.add_subplot(222)

    sigma_sfr = []
    sigma_sfr_err = []
    for mb in mass_bins:
        idx = (mass > (mb-deltamass)) & (mass <= (mb+deltamass))
        sigma_sfr.append(np.std(sfr[idx]))
        try:
            sigma_sfr_err.append(1./np.sum(idx))
        except ZeroDivisionError:
            sigma_sfr_err.append(0.)

    ax.errorbar(mass_bins,sigma_sfr,yerr=sigma_sfr_err,fmt='o',color='black',capsize=0)

    arm_tasks = ('a28_tight','a29_medium','a30_loose')
    colors = ('red','green','blue')
    for idx, (a,c) in enumerate(zip(arm_tasks,colors)):

        spiral = (sf_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sf_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sf_sample['t04_spiral_a08_spiral_weight'] >= 20) & (sf_sample['t04_spiral_a08_spiral_debiased'] > 0.619)
        morph_sample = sf_sample[spiral & (sf_sample['t10_arms_winding_%s_flag' % a] == 1)] 

        sigma_sfr,sigma_sfr_err,ngals = [],[],[]
        for mb in mass_bins:
            idx = (morph_sample['MEDIAN_MASS'] > (mb-deltamass)) & (morph_sample['MEDIAN_MASS'] <= (mb+deltamass))
            sigma_sfr.append(np.std(morph_sample['MEDIAN_SFR'][idx]))
            try:
                sigma_sfr_err.append(1./np.sum(idx))
            except ZeroDivisionError:
                sigma_sfr_err.append(0.)
            ngals.append(np.sum(idx))

        gtr10 = (np.array(ngals) >= 10)
        ax.errorbar(mass_bins[gtr10],np.array(sigma_sfr)[gtr10],yerr=np.array(sigma_sfr_err)[gtr10],color=c,capsize=0)

    # Final plot labels and ranges
    ax.set_xlim(8,11.5)
    ax.set_ylim(0,1.2)
    ax.set_ylabel(r'$\sigma_{SFR}~[\log M_\odot/\mathrm{yr}]$',fontsize=16)
    ax.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',fontsize=16)
    ax.set_title('Arms winding')

    # Mergers
    ax = fig.add_subplot(224)

    sigma_sfr = []
    sigma_sfr_err = []
    for mb in mass_bins:
        idx = (mass > (mb-deltamass)) & (mass <= (mb+deltamass))
        sigma_sfr.append(np.std(sfr[idx]))
        try:
            sigma_sfr_err.append(1./np.sum(idx))
        except ZeroDivisionError:
            sigma_sfr_err.append(0.)

    ax.errorbar(mass_bins,sigma_sfr,yerr=sigma_sfr_err,fmt='o',color='black',capsize=0)

    with fits.open('%s/mergers/mergers_mpajhu_bpt.fits' % gzm_path) as f:
        data = f[1].data

    sf_mergers = data[(data['bpt'] == 1)]

    sigma_sfr,sigma_sfr_err,ngals = [],[],[]
    for mb in mass_bins:
        idx = (sf_mergers['MEDIAN_MASS'] > (mb-deltamass)) & (sf_mergers['MEDIAN_MASS'] <= (mb+deltamass))
        sigma_sfr.append(np.std(sf_mergers['MEDIAN_SFR'][idx]))
        try:
            sigma_sfr_err.append(1./np.sum(idx))
        except ZeroDivisionError:
            sigma_sfr_err.append(0.)
        ngals.append(np.sum(idx))

    gtr10 = (np.array(ngals) >= 10)
    ax.errorbar(mass_bins[gtr10],np.array(sigma_sfr)[gtr10],yerr=np.array(sigma_sfr_err)[gtr10],color='red',capsize=0)

    # Final plot labels and ranges
    ax.set_xlim(8,11.5)
    ax.set_ylim(0,1.2)
    ax.set_ylabel(r'$\sigma_{SFR}~[\log M_\odot/\mathrm{yr}]$',fontsize=16)
    ax.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',fontsize=16)
    ax.set_title('Mergers')

    fig.set_tight_layout(True)
    fig.savefig('%s/sigma_mstar.pdf' % fig_path, dpi=200)

    return None

def plot_ms_mergers_both(sf_sample,plot_ssfr=False,verbose=False):

    #---------------------------------
    # Merger counts
    #---------------------------------

    fig = plt.figure(8,(14,7))
    fig.clf()
    filestr=''

    #fig.subplots_adjust(left=0.08,bottom=0.15,right=0.90,hspace=0,wspace=0.05)

    mass,sfr,sfr_err,ssfr,ssfr_err = get_mass_sfr(sf_sample)

    # Get mergers data

    with fits.open('%s/mergers/mergers_mpajhu_bpt.fits' % gzm_path) as f:
        data = f[1].data

    sf_mergers = data[(data['bpt'] == 1) & (data['MEDIAN_MASS'] > 8.)]

    if plot_ssfr:
        ylim1 = (-12,-8)
        ylabel='sSFR '+r'$[\log\/\mathrm{yr}^{-1}]$'
        yval = ssfr
        yval_mergers = sf_mergers['MEDIAN_SFR'] - sf_mergers['MEDIAN_MASS']
        filestr += '_ssfr'
    else:
        ylim1 = (-3,2)
        ylabel='SFR '+r'$[\log\/M_\odot/\mathrm{yr}]$'
        yval = sfr
        yval_mergers = sf_mergers['MEDIAN_SFR']

    # Plot star-forming galaxies

    ax1 = fig.add_axes([0.05,0.10,0.42,0.85])
    h = ax1.hist2d(mass,yval,bins=50,cmap = cm.gray_r, norm=LogNorm())
    ax1.set_xlim(8,11.5)
    ax1.set_ylim(ylim1)
    ax1.set_ylabel(ylabel,fontsize=16)
    ax1.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',fontsize=16)

    sc = ax1.scatter(sf_mergers['MEDIAN_MASS'],yval_mergers, c=sf_mergers['mass_ratio'], edgecolor='none',s = 50, marker='.', cmap=cm.RdBu, vmin=1.,vmax=10.)

    # Plot the best linear fit

    a1,a0 = plot_fits('Star-forming galaxies',mass,yval,None,ax1,'black',lw=3,ls='-',verbose=verbose)

    # Fix same slope but different offset. 

    mass_mergers_plot,sfr_mergers_plot,sfr_err_mergers_plot,ssfr_mergers_plot,ssfr_err_mergers_plot = get_mass_sfr(sf_mergers)

    if plot_ssfr:
        yval_mergers_plot = ssfr_mergers_plot
        ytextloc = (-11.0,-11.5)
    else:
        yval_mergers_plot = sfr_mergers_plot
        ytextloc = (1.5,1.2)

    rms = []
    xarr = np.linspace(-1,1,200)
    for x in xarr:
        rms.append(np.sqrt(np.sum((yval_mergers_plot - (a0 + x + a1*mass_mergers_plot))**2)))

    offset = xarr[(np.abs(rms)).argmin()]
    uline = ax1.plot(np.linspace(6,12,100),np.polyval([a1,a0+offset],np.linspace(6,12,100)),color='red',linestyle='--',linewidth=3)

    if verbose:
        print 'Best fit for mergers data: a,b = %.2f,%.2f' % (a1,a0 + offset)
        print 'Best fit for mergers data: same sigma_a, sigma_b as star-forming galaxies'
        print ''

    ax1.text(8.1,ytextloc[0],'Mergers', color='black',fontsize=18)
    ax1.text(8.1,ytextloc[1],'Offset = %.3f dex' % offset, color='black',fontsize=13)

    # Set the colorbars and labels

    axcb1 = plt.axes([0.38, 0.20, 0.03, 0.25]) 
    cb1 = plt.colorbar(sc,cax = axcb1, orientation="vertical",ticks=[1,3,5,10])
    cb1.set_label('Mass ratio',fontsize=14)

    ax1.xaxis.labelpad = 15
    #---------------------------------
    # Merger fraction
    #---------------------------------

    mass_bins,sfr_bins,ssfr_bins = bins(nbins=30)

    if plot_ssfr:
        yval_bins = ssfr_bins
    else:
        yval_bins = sfr_bins

    h,xedges,yedges = np.histogram2d(mass,yval,bins=(mass_bins,yval_bins))

    # Plot star-forming galaxies

    ax2 = fig.add_axes([0.458,0.10,0.45,0.85])
    levels=10**(np.linspace(0,5,15))
    CS = ax2.contour(mass_bins[1:],yval_bins[1:],h.T,levels,colors='black')
    ax2.set_xlim(8,11.5)
    ax2.set_ylim(ylim1)
    ax2.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',fontsize=16)

    hm,xm,ym = np.histogram2d(sf_mergers['MEDIAN_MASS'],yval_mergers,bins=(mass_bins,yval_bins))
    # Calculate merger fraction
    mf = hm / h
    im = ax2.imshow(mf.T, interpolation='nearest', origin='lower',extent=(mass_bins[0],mass_bins[-1],yval_bins[0],yval_bins[-1]),vmin=0.,vmax=1.,cmap=cm.jet)

    axcb2 = plt.axes([0.91, 0.10, 0.02, 0.85]) 
    cb2 = plt.colorbar(im,cax = axcb2, orientation="vertical")
    cb2.set_label('Merger fraction',fontsize=16)

    ax2.text(8.1,ytextloc[0],'Merger fraction', color='black',fontsize=18)
    if plot_ssfr:
        ax2.set_aspect('auto')
    ax2.set_aspect(0.71)

    ax2.set_xticklabels([' ',8.5,9.0,9.5,10.0,10.5,11.0,11.5])
    ax2.set_yticklabels([])

    # Plot the best linear fits

    a1,a0 = plot_fits('Star-forming galaxies',mass,yval,None,ax2,'black',lw=3,ls='-')
    uline = ax2.plot(np.linspace(6,12,100),np.polyval([a1,a0+offset],np.linspace(6,12,100)),color='red',linestyle='--',linewidth=3)

    ax2.xaxis.labelpad = 15

    #fig.set_tight_layout(True)
    fig.savefig('%s/ms_mergers_both%s.pdf' % (fig_path,filestr), dpi=200)

    return None

def fracdev(sf_sample):

    # Find the fracdev distribution for all of the various spiral classes, per Ivy Wong's suggestion

    # Contained in the original gz2_sample table, and thus in the joined mpajhu_gz2.fits table.

    # Overall sample, plus 6 multiplicity, 3 winding, 2 bars, merger = 12

    # Good opportunity to take the plurality answer

    fig = plt.figure(9,(15,10))
    fig.clf()

    fig.subplots_adjust(left=0.08,bottom=0.15,right=0.9,hspace=0.02,wspace=0.02)

    spiral = (sf_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sf_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sf_sample['t04_spiral_a08_spiral_weight'] >= 20) & (sf_sample['t04_spiral_a08_spiral_debiased'] > 0.619)

    # Find plurality answer for multiplicity question

    an_1 = sf_sample[spiral & \
                       (sf_sample['t11_arms_number_a31_1_debiased'] >= sf_sample['t11_arms_number_a32_2_debiased']) & \
                       (sf_sample['t11_arms_number_a31_1_debiased'] >= sf_sample['t11_arms_number_a33_3_debiased']) & \
                       (sf_sample['t11_arms_number_a31_1_debiased'] >= sf_sample['t11_arms_number_a34_4_debiased']) & \
                       (sf_sample['t11_arms_number_a31_1_debiased'] >= sf_sample['t11_arms_number_a36_more_than_4_debiased']) & \
                       (sf_sample['t11_arms_number_a31_1_debiased'] >= sf_sample['t11_arms_number_a37_cant_tell_debiased']) ]

    an_2 = sf_sample[spiral & \
                       (sf_sample['t11_arms_number_a32_2_debiased'] >= sf_sample['t11_arms_number_a31_1_debiased']) & \
                       (sf_sample['t11_arms_number_a32_2_debiased'] >= sf_sample['t11_arms_number_a33_3_debiased']) & \
                       (sf_sample['t11_arms_number_a32_2_debiased'] >= sf_sample['t11_arms_number_a34_4_debiased']) & \
                       (sf_sample['t11_arms_number_a32_2_debiased'] >= sf_sample['t11_arms_number_a36_more_than_4_debiased']) & \
                       (sf_sample['t11_arms_number_a32_2_debiased'] >= sf_sample['t11_arms_number_a37_cant_tell_debiased']) ]

    an_3 = sf_sample[spiral & \
                       (sf_sample['t11_arms_number_a33_3_debiased'] >= sf_sample['t11_arms_number_a32_2_debiased']) & \
                       (sf_sample['t11_arms_number_a33_3_debiased'] >= sf_sample['t11_arms_number_a31_1_debiased']) & \
                       (sf_sample['t11_arms_number_a33_3_debiased'] >= sf_sample['t11_arms_number_a34_4_debiased']) & \
                       (sf_sample['t11_arms_number_a33_3_debiased'] >= sf_sample['t11_arms_number_a36_more_than_4_debiased']) & \
                       (sf_sample['t11_arms_number_a33_3_debiased'] >= sf_sample['t11_arms_number_a37_cant_tell_debiased']) ]

    an_4 = sf_sample[spiral & \
                       (sf_sample['t11_arms_number_a34_4_debiased'] >= sf_sample['t11_arms_number_a32_2_debiased']) & \
                       (sf_sample['t11_arms_number_a34_4_debiased'] >= sf_sample['t11_arms_number_a33_3_debiased']) & \
                       (sf_sample['t11_arms_number_a34_4_debiased'] >= sf_sample['t11_arms_number_a31_1_debiased']) & \
                       (sf_sample['t11_arms_number_a34_4_debiased'] >= sf_sample['t11_arms_number_a36_more_than_4_debiased']) & \
                       (sf_sample['t11_arms_number_a34_4_debiased'] >= sf_sample['t11_arms_number_a37_cant_tell_debiased']) ]

    an_5 = sf_sample[spiral & \
                       (sf_sample['t11_arms_number_a36_more_than_4_debiased'] >= sf_sample['t11_arms_number_a32_2_debiased']) & \
                       (sf_sample['t11_arms_number_a36_more_than_4_debiased'] >= sf_sample['t11_arms_number_a33_3_debiased']) & \
                       (sf_sample['t11_arms_number_a36_more_than_4_debiased'] >= sf_sample['t11_arms_number_a34_4_debiased']) & \
                       (sf_sample['t11_arms_number_a36_more_than_4_debiased'] >= sf_sample['t11_arms_number_a31_1_debiased']) & \
                       (sf_sample['t11_arms_number_a36_more_than_4_debiased'] >= sf_sample['t11_arms_number_a37_cant_tell_debiased']) ]

    an_x = sf_sample[spiral & \
                       (sf_sample['t11_arms_number_a37_cant_tell_debiased'] >= sf_sample['t11_arms_number_a32_2_debiased']) & \
                       (sf_sample['t11_arms_number_a37_cant_tell_debiased'] >= sf_sample['t11_arms_number_a33_3_debiased']) & \
                       (sf_sample['t11_arms_number_a37_cant_tell_debiased'] >= sf_sample['t11_arms_number_a34_4_debiased']) & \
                       (sf_sample['t11_arms_number_a37_cant_tell_debiased'] >= sf_sample['t11_arms_number_a36_more_than_4_debiased']) & \
                       (sf_sample['t11_arms_number_a37_cant_tell_debiased'] >= sf_sample['t11_arms_number_a31_1_debiased']) ]

    aw_t = sf_sample[spiral & \
                       (sf_sample['t10_arms_winding_a28_tight_debiased'] >= sf_sample['t10_arms_winding_a29_medium_debiased']) & \
                       (sf_sample['t10_arms_winding_a28_tight_debiased'] >= sf_sample['t10_arms_winding_a30_loose_debiased'])]

    aw_m = sf_sample[spiral & \
                       (sf_sample['t10_arms_winding_a29_medium_debiased'] >= sf_sample['t10_arms_winding_a28_tight_debiased']) & \
                       (sf_sample['t10_arms_winding_a29_medium_debiased'] >= sf_sample['t10_arms_winding_a30_loose_debiased'])]

    aw_l = sf_sample[spiral & \
                       (sf_sample['t10_arms_winding_a30_loose_debiased'] >= sf_sample['t10_arms_winding_a29_medium_debiased']) & \
                       (sf_sample['t10_arms_winding_a30_loose_debiased'] >= sf_sample['t10_arms_winding_a28_tight_debiased'])]

    notedgeon = (sf_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sf_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sf_sample['t02_edgeon_a05_no_weight'] >= 20)

    barred = sf_sample[notedgeon & (sf_sample['t03_bar_a06_bar_debiased'] >= 0.5)] 
    unbarred = sf_sample[notedgeon & (sf_sample['t03_bar_a06_bar_debiased'] < 0.5)] 

    # Print stats to screen

    print 'All SF galaxies: %10i' % len(sf_sample)
    print 'Arms number'
    print ' 1 arm: %10i' % len(an_1)
    print ' 2 arm: %10i' % len(an_2)
    print ' 3 arm: %10i' % len(an_3)
    print ' 4 arm: %10i' % len(an_4)
    print '5+ arm: %10i' % len(an_5)
    print '?? arm: %10i' % len(an_x)
    print ''
    print 'Arms winding'
    print ' tight arm: %10i' % len(aw_t)
    print 'medium arm: %10i' % len(aw_m)
    print ' loose arm: %10i' % len(aw_l)
    print ''
    print 'Bar'
    print '  barred: %10i' % len(barred)
    print 'unbarred: %10i' % len(unbarred)

    with fits.open('%s/mergers/mergers_fracdev.fits' % gzm_path) as f:
        data = f[1].data

    sf_mergers = data[(data['bpt'] == 1)]

    axc = fig.add_subplot(111)    # The big subplot
    axc.spines['top'].set_color('none')
    axc.spines['bottom'].set_color('none')
    axc.spines['left'].set_color('none')
    axc.spines['right'].set_color('none')
    axc.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    axc.set_xlabel(r'$f_{DeV}$ (r-band)',labelpad=10,fontsize=20)
    axc.set_ylabel('Normalized count',labelpad=20,fontsize=20)

    axc.set_title('frac deV distribution for GZ2 galaxies',fontsize=25)

    ax = fig.add_subplot(341)
    n, bins, patches = ax.hist(sf_sample['FRACDEV_R'], 25, histtype='stepfilled', weights=np.ones_like(sf_sample['FRACDEV_R'])/len(sf_sample), facecolor='gray', edgecolor='black', alpha=0.75)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([0.0,0.10,0.20,0.30])
    n_sub, bins_sub, patches_sub = ax.hist(an_1['FRACDEV_R'], 25, histtype='step', weights=np.ones_like(an_1['FRACDEV_R'])/len(an_1), edgecolor=an_colors[0], alpha=1., lw=3)
    ax.text(0.95,0.28,'1 arm',fontsize=12,ha='right')

    ax = fig.add_subplot(342)
    n, bins, patches = ax.hist(sf_sample['FRACDEV_R'], 25, histtype='stepfilled', weights=np.ones_like(sf_sample['FRACDEV_R'])/len(sf_sample), facecolor='gray', edgecolor='black', alpha=0.75)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    n_sub, bins_sub, patches_sub = ax.hist(an_2['FRACDEV_R'], 25, histtype='step', weights=np.ones_like(an_2['FRACDEV_R'])/len(an_2), edgecolor=an_colors[1], alpha=1., lw=3)
    ax.text(0.95,0.28,'2 arms',fontsize=12,ha='right')

    ax = fig.add_subplot(343)
    n, bins, patches = ax.hist(sf_sample['FRACDEV_R'], 25, histtype='stepfilled', weights=np.ones_like(sf_sample['FRACDEV_R'])/len(sf_sample), facecolor='gray', edgecolor='black', alpha=0.75)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    n_sub, bins_sub, patches_sub = ax.hist(an_3['FRACDEV_R'], 25, histtype='step', weights=np.ones_like(an_3['FRACDEV_R'])/len(an_3), edgecolor=an_colors[2], alpha=1., lw=3)
    ax.text(0.95,0.28,'3 arms',fontsize=12,ha='right')

    ax = fig.add_subplot(344)
    n, bins, patches = ax.hist(sf_sample['FRACDEV_R'], 25, histtype='stepfilled', weights=np.ones_like(sf_sample['FRACDEV_R'])/len(sf_sample), facecolor='gray', edgecolor='black', alpha=0.75)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    n_sub, bins_sub, patches_sub = ax.hist(an_4['FRACDEV_R'], 25, histtype='step', weights=np.ones_like(an_4['FRACDEV_R'])/len(an_4), edgecolor=an_colors[3], alpha=1., lw=3)
    ax.text(0.95,0.28,'4 arms',fontsize=12,ha='right')

    ax = fig.add_subplot(345)
    n, bins, patches = ax.hist(sf_sample['FRACDEV_R'], 25, histtype='stepfilled', weights=np.ones_like(sf_sample['FRACDEV_R'])/len(sf_sample), facecolor='gray', edgecolor='black', alpha=0.75)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([0.0,0.10,0.20,0.30])
    n_sub, bins_sub, patches_sub = ax.hist(an_5['FRACDEV_R'], 25, histtype='step', weights=np.ones_like(an_5['FRACDEV_R'])/len(an_5), edgecolor=an_colors[4], alpha=1., lw=3)
    ax.text(0.95,0.28,'5+ arms',fontsize=12,ha='right')

    ax = fig.add_subplot(346)
    n, bins, patches = ax.hist(sf_sample['FRACDEV_R'], 25, histtype='stepfilled', weights=np.ones_like(sf_sample['FRACDEV_R'])/len(sf_sample), facecolor='gray', edgecolor='black', alpha=0.75)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    n_sub, bins_sub, patches_sub = ax.hist(an_x['FRACDEV_R'], 25, histtype='step', weights=np.ones_like(an_x['FRACDEV_R'])/len(an_x), edgecolor=an_colors[5], alpha=1., lw=3)
    ax.text(0.95,0.28,'?? arms',fontsize=12,ha='right')

    ax = fig.add_subplot(347)
    n, bins, patches = ax.hist(sf_sample['FRACDEV_R'], 25, histtype='stepfilled', weights=np.ones_like(sf_sample['FRACDEV_R'])/len(sf_sample), facecolor='gray', edgecolor='black', alpha=0.75)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    n_sub, bins_sub, patches_sub = ax.hist(aw_t['FRACDEV_R'], 25, histtype='step', weights=np.ones_like(aw_t['FRACDEV_R'])/len(aw_t), edgecolor=aw_colors[0], alpha=1., lw=3)
    ax.text(0.95,0.28,'Tight arms',fontsize=12,ha='right')

    ax = fig.add_subplot(348)
    n, bins, patches = ax.hist(sf_sample['FRACDEV_R'], 25, histtype='stepfilled', weights=np.ones_like(sf_sample['FRACDEV_R'])/len(sf_sample), facecolor='gray', edgecolor='black', alpha=0.75)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    n_sub, bins_sub, patches_sub = ax.hist(aw_m['FRACDEV_R'], 25, histtype='step', weights=np.ones_like(aw_m['FRACDEV_R'])/len(aw_m), edgecolor=aw_colors[1], alpha=1., lw=3)
    ax.text(0.95,0.28,'Medium arms',fontsize=12,ha='right')

    ax = fig.add_subplot(349)
    n, bins, patches = ax.hist(sf_sample['FRACDEV_R'], 25, histtype='stepfilled', weights=np.ones_like(sf_sample['FRACDEV_R'])/len(sf_sample), facecolor='gray', edgecolor='black', alpha=0.75)
    ax.get_xaxis().set_ticks([0.0,0.2,0.4,0.6,0.8])
    ax.get_yaxis().set_ticks([0.0,0.10,0.20,0.30])
    n_sub, bins_sub, patches_sub = ax.hist(aw_l['FRACDEV_R'], 25, histtype='step', weights=np.ones_like(aw_l['FRACDEV_R'])/len(aw_l), edgecolor=aw_colors[2], alpha=1., lw=3)
    ax.text(0.95,0.28,'Loose arms',fontsize=12,ha='right')

    ax = fig.add_subplot(3,4,10)
    n, bins, patches = ax.hist(sf_sample['FRACDEV_R'], 25, histtype='stepfilled', weights=np.ones_like(sf_sample['FRACDEV_R'])/len(sf_sample), facecolor='gray', edgecolor='black', alpha=0.75)
    ax.get_xaxis().set_ticks([0.0,0.2,0.4,0.6,0.8])
    ax.get_yaxis().set_ticks([])
    n_sub, bins_sub, patches_sub = ax.hist(unbarred['FRACDEV_R'], 25, histtype='step', weights=np.ones_like(unbarred['FRACDEV_R'])/len(unbarred), edgecolor='red', alpha=1., lw=3)
    ax.text(0.95,0.28,'Unbarred',fontsize=12,ha='right')

    ax = fig.add_subplot(3,4,11)
    n, bins, patches = ax.hist(sf_sample['FRACDEV_R'], 25, histtype='stepfilled', weights=np.ones_like(sf_sample['FRACDEV_R'])/len(sf_sample), facecolor='gray', edgecolor='black', alpha=0.75)
    ax.get_xaxis().set_ticks([0.0,0.2,0.4,0.6,0.8])
    ax.get_yaxis().set_ticks([])
    n_sub, bins_sub, patches_sub = ax.hist(barred['FRACDEV_R'], 25, histtype='step', weights=np.ones_like(barred['FRACDEV_R'])/len(barred), edgecolor='blue', alpha=1., lw=3)
    ax.text(0.95,0.28,'Barred',fontsize=12,ha='right')

    ax = fig.add_subplot(3,4,12)
    n, bins, patches = ax.hist(sf_sample['FRACDEV_R'], 25, histtype='stepfilled', weights=np.ones_like(sf_sample['FRACDEV_R'])/len(sf_sample), facecolor='gray', edgecolor='black', alpha=0.75)
    ax.get_yaxis().set_ticks([])
    n_sub, bins_sub, patches_sub = ax.hist(sf_mergers['FRACDEV_R'], 25, histtype='step', weights=np.ones_like(sf_mergers['FRACDEV_R'])/len(sf_mergers), edgecolor='red', alpha=1., lw=3)
    ax.text(0.95,0.28,'Mergers',fontsize=12,ha='right')

    plt.show()

    fig.savefig('%s/fracdev_hist.pdf' % fig_path, dpi=200)

    return None

def fracdev_cdf(sf_sample):

    # Find the CDF of the fracdev distribution for all of the various spiral classes, per Ivy Wong's suggestion

    fig = plt.figure(10,(10,10))
    fig.clf()

    fig.subplots_adjust(left=0.08,bottom=0.15,right=0.9,hspace=0.02,wspace=0.02)

    spiral = (sf_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sf_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sf_sample['t04_spiral_a08_spiral_weight'] >= 20) & (sf_sample['t04_spiral_a08_spiral_debiased'] > 0.619)

    # Find plurality answer for multiplicity question

    an_1 = sf_sample[spiral & \
                       (sf_sample['t11_arms_number_a31_1_debiased'] >= sf_sample['t11_arms_number_a32_2_debiased']) & \
                       (sf_sample['t11_arms_number_a31_1_debiased'] >= sf_sample['t11_arms_number_a33_3_debiased']) & \
                       (sf_sample['t11_arms_number_a31_1_debiased'] >= sf_sample['t11_arms_number_a34_4_debiased']) & \
                       (sf_sample['t11_arms_number_a31_1_debiased'] >= sf_sample['t11_arms_number_a36_more_than_4_debiased']) & \
                       (sf_sample['t11_arms_number_a31_1_debiased'] >= sf_sample['t11_arms_number_a37_cant_tell_debiased']) ]

    an_2 = sf_sample[spiral & \
                       (sf_sample['t11_arms_number_a32_2_debiased'] >= sf_sample['t11_arms_number_a31_1_debiased']) & \
                       (sf_sample['t11_arms_number_a32_2_debiased'] >= sf_sample['t11_arms_number_a33_3_debiased']) & \
                       (sf_sample['t11_arms_number_a32_2_debiased'] >= sf_sample['t11_arms_number_a34_4_debiased']) & \
                       (sf_sample['t11_arms_number_a32_2_debiased'] >= sf_sample['t11_arms_number_a36_more_than_4_debiased']) & \
                       (sf_sample['t11_arms_number_a32_2_debiased'] >= sf_sample['t11_arms_number_a37_cant_tell_debiased']) ]

    an_3 = sf_sample[spiral & \
                       (sf_sample['t11_arms_number_a33_3_debiased'] >= sf_sample['t11_arms_number_a32_2_debiased']) & \
                       (sf_sample['t11_arms_number_a33_3_debiased'] >= sf_sample['t11_arms_number_a31_1_debiased']) & \
                       (sf_sample['t11_arms_number_a33_3_debiased'] >= sf_sample['t11_arms_number_a34_4_debiased']) & \
                       (sf_sample['t11_arms_number_a33_3_debiased'] >= sf_sample['t11_arms_number_a36_more_than_4_debiased']) & \
                       (sf_sample['t11_arms_number_a33_3_debiased'] >= sf_sample['t11_arms_number_a37_cant_tell_debiased']) ]

    an_4 = sf_sample[spiral & \
                       (sf_sample['t11_arms_number_a34_4_debiased'] >= sf_sample['t11_arms_number_a32_2_debiased']) & \
                       (sf_sample['t11_arms_number_a34_4_debiased'] >= sf_sample['t11_arms_number_a33_3_debiased']) & \
                       (sf_sample['t11_arms_number_a34_4_debiased'] >= sf_sample['t11_arms_number_a31_1_debiased']) & \
                       (sf_sample['t11_arms_number_a34_4_debiased'] >= sf_sample['t11_arms_number_a36_more_than_4_debiased']) & \
                       (sf_sample['t11_arms_number_a34_4_debiased'] >= sf_sample['t11_arms_number_a37_cant_tell_debiased']) ]

    an_5 = sf_sample[spiral & \
                       (sf_sample['t11_arms_number_a36_more_than_4_debiased'] >= sf_sample['t11_arms_number_a32_2_debiased']) & \
                       (sf_sample['t11_arms_number_a36_more_than_4_debiased'] >= sf_sample['t11_arms_number_a33_3_debiased']) & \
                       (sf_sample['t11_arms_number_a36_more_than_4_debiased'] >= sf_sample['t11_arms_number_a34_4_debiased']) & \
                       (sf_sample['t11_arms_number_a36_more_than_4_debiased'] >= sf_sample['t11_arms_number_a31_1_debiased']) & \
                       (sf_sample['t11_arms_number_a36_more_than_4_debiased'] >= sf_sample['t11_arms_number_a37_cant_tell_debiased']) ]

    an_x = sf_sample[spiral & \
                       (sf_sample['t11_arms_number_a37_cant_tell_debiased'] >= sf_sample['t11_arms_number_a32_2_debiased']) & \
                       (sf_sample['t11_arms_number_a37_cant_tell_debiased'] >= sf_sample['t11_arms_number_a33_3_debiased']) & \
                       (sf_sample['t11_arms_number_a37_cant_tell_debiased'] >= sf_sample['t11_arms_number_a34_4_debiased']) & \
                       (sf_sample['t11_arms_number_a37_cant_tell_debiased'] >= sf_sample['t11_arms_number_a36_more_than_4_debiased']) & \
                       (sf_sample['t11_arms_number_a37_cant_tell_debiased'] >= sf_sample['t11_arms_number_a31_1_debiased']) ]

    aw_t = sf_sample[spiral & \
                       (sf_sample['t10_arms_winding_a28_tight_debiased'] >= sf_sample['t10_arms_winding_a29_medium_debiased']) & \
                       (sf_sample['t10_arms_winding_a28_tight_debiased'] >= sf_sample['t10_arms_winding_a30_loose_debiased'])]

    aw_m = sf_sample[spiral & \
                       (sf_sample['t10_arms_winding_a29_medium_debiased'] >= sf_sample['t10_arms_winding_a28_tight_debiased']) & \
                       (sf_sample['t10_arms_winding_a29_medium_debiased'] >= sf_sample['t10_arms_winding_a30_loose_debiased'])]

    aw_l = sf_sample[spiral & \
                       (sf_sample['t10_arms_winding_a30_loose_debiased'] >= sf_sample['t10_arms_winding_a29_medium_debiased']) & \
                       (sf_sample['t10_arms_winding_a30_loose_debiased'] >= sf_sample['t10_arms_winding_a28_tight_debiased'])]

    aw_l = sf_sample[spiral & \
                       (sf_sample['t10_arms_winding_a30_loose_debiased'] >= sf_sample['t10_arms_winding_a29_medium_debiased']) & \
                       (sf_sample['t10_arms_winding_a30_loose_debiased'] >= sf_sample['t10_arms_winding_a28_tight_debiased'])]


    notedgeon = (sf_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sf_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sf_sample['t02_edgeon_a05_no_weight'] >= 20)

    barred = sf_sample[notedgeon & (sf_sample['t03_bar_a06_bar_debiased'] >= 0.5)] 
    unbarred = sf_sample[notedgeon & (sf_sample['t03_bar_a06_bar_debiased'] < 0.5)] 

    with fits.open('%s/mergers/mergers_fracdev.fits' % gzm_path) as f:
        data = f[1].data

    sf_mergers = data[(data['bpt'] == 1)]

    axc = fig.add_subplot(111)    # The big subplot
    axc.spines['top'].set_color('none')
    axc.spines['bottom'].set_color('none')
    axc.spines['left'].set_color('none')
    axc.spines['right'].set_color('none')
    axc.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    axc.set_xlabel(r'$f_{DeV}$ (r-band)',labelpad=10,fontsize=20)
    axc.set_ylabel('Normalized cumulative sum',labelpad=10,fontsize=20)

    axc.set_title('frac deV CDF for GZ2 galaxies',fontsize=25)

    nbins = 20

    ax = fig.add_subplot(221)
    n, bins, patches = ax.hist(sf_sample['FRACDEV_R'], nbins, histtype='step', cumulative=True, normed=True, edgecolor='black', alpha=1.0, lw=3)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks(np.arange(6)*0.2)
    n_sub, bins_sub, patches_sub = ax.hist(an_1['FRACDEV_R'], nbins, histtype='step', cumulative=True, normed=True, edgecolor=an_colors[0], alpha=1.0, lw=1)
    n_sub, bins_sub, patches_sub = ax.hist(an_2['FRACDEV_R'], nbins, histtype='step', cumulative=True, normed=True, edgecolor=an_colors[1], alpha=1.0, lw=1)
    n_sub, bins_sub, patches_sub = ax.hist(an_3['FRACDEV_R'], nbins, histtype='step', cumulative=True, normed=True, edgecolor=an_colors[2], alpha=1.0, lw=1)
    n_sub, bins_sub, patches_sub = ax.hist(an_4['FRACDEV_R'], nbins, histtype='step', cumulative=True, normed=True, edgecolor=an_colors[3], alpha=1.0, lw=1)
    n_sub, bins_sub, patches_sub = ax.hist(an_5['FRACDEV_R'], nbins, histtype='step', cumulative=True, normed=True, edgecolor=an_colors[4], alpha=1.0, lw=1)
    n_sub, bins_sub, patches_sub = ax.hist(an_x['FRACDEV_R'], nbins, histtype='step', cumulative=True, normed=True, edgecolor=an_colors[5], alpha=1.0, lw=1)
    ax.text(0.50,0.40,'Arms number',fontsize=16,ha='left')

    ax = fig.add_subplot(222)
    n, bins, patches = ax.hist(sf_sample['FRACDEV_R'], nbins, histtype='step', cumulative=True, normed=True, edgecolor='black', alpha=1.0, lw=3)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    n_sub, bins_sub, patches_sub = ax.hist(aw_t['FRACDEV_R'], nbins, histtype='step', cumulative=True, normed=True, edgecolor=aw_colors[0], alpha=1.0, lw=1)
    n_sub, bins_sub, patches_sub = ax.hist(aw_m['FRACDEV_R'], nbins, histtype='step', cumulative=True, normed=True, edgecolor=aw_colors[1], alpha=1.0, lw=1)
    n_sub, bins_sub, patches_sub = ax.hist(aw_l['FRACDEV_R'], nbins, histtype='step', cumulative=True, normed=True, edgecolor=aw_colors[2], alpha=1.0, lw=1)
    ax.text(0.50,0.40,'Arms winding',fontsize=16,ha='left')

    ax = fig.add_subplot(223)
    ax.get_xaxis().set_ticks(np.arange(5)*0.2)
    ax.get_yaxis().set_ticks(np.arange(5)*0.2)
    n, bins, patches = ax.hist(sf_sample['FRACDEV_R'], nbins, histtype='step', cumulative=True, normed=True, edgecolor='black', alpha=1.0, lw=3)
    n_sub, bins_sub, patches_sub = ax.hist(barred['FRACDEV_R'], nbins, histtype='step', cumulative=True, normed=True, edgecolor='blue', alpha=1.0, lw=1)
    n_sub, bins_sub, patches_sub = ax.hist(unbarred['FRACDEV_R'], nbins, histtype='step', cumulative=True, normed=True, edgecolor='red', alpha=1.0, lw=1)
    ax.text(0.50,0.40,'Bars',fontsize=16,ha='left')

    ax = fig.add_subplot(224)
    ax.get_xaxis().set_ticks(np.arange(6)*0.2)
    ax.get_yaxis().set_ticks([])
    n, bins, patches = ax.hist(sf_sample['FRACDEV_R'], nbins, histtype='step', cumulative=True, normed=True, edgecolor='black', alpha=1.0, lw=3)
    n_sub, bins_sub, patches_sub = ax.hist(sf_mergers['FRACDEV_R'], nbins, histtype='step', cumulative=True, normed=True, edgecolor='red', alpha=1.0, lw=1)
    ax.text(0.50,0.40,'Mergers',fontsize=16,ha='left')

    plt.show()

    fig.savefig('%s/fracdev_cdf.pdf' % fig_path, dpi=200)

    return None

def tm_bar_trigger(sf_sample):

    '''
    Idea from Tom Melvin:

    I would be interested to see what the relationships for 1 armed spirals and merging galaxies would look like for barred and unbarred versions of the two. As there is the idea that mergers can induce bars, do merging disk galaxies with bars deviate further from the SFMS or not? I'm not sure of the numbers for each of the samples once plit into barred and unbarred, but it will probably be dependent of mass ratios (as you use in Fig 4), as 1-to-1 mergers would probably have no bars!
    '''

    # Plot

    fig = plt.figure(11,(12,10))
    fig.clf()

    fig.subplots_adjust(left=0.08,bottom=0.15,right=0.9,hspace=0,wspace=0)

    mass,sfr,sfr_err,ssfr,ssfr_err = get_mass_sfr(sf_sample)
    mass_bins,sfr_bins,ssfr_bins = bins()
    h,xedges,yedges = np.histogram2d(mass,sfr,bins=(mass_bins,sfr_bins))

    # Plot each one

    responses = ('a31_1','a32_2','a33_3','a34_4','a36_more_than_4','a37_cant_tell')
    arm_tasks = tuple(['t11_arms_number_'+r for r in responses])
    arm_label = ('1','2','3','4','5+','??')
    colors = ('red','orange','yellow','green','blue','purple')

    # Find 1-armed galaxies, then split into barred and unbarred

    spiral = (sf_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sf_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sf_sample['t04_spiral_a08_spiral_weight'] >= 20) & (sf_sample['t04_spiral_a08_spiral_debiased'] > 0.619)
    an_1 = (sf_sample['t11_arms_number_a31_1_debiased'] >= sf_sample['t11_arms_number_a32_2_debiased']) & \
                (sf_sample['t11_arms_number_a31_1_debiased'] >= sf_sample['t11_arms_number_a33_3_debiased']) & \
                (sf_sample['t11_arms_number_a31_1_debiased'] >= sf_sample['t11_arms_number_a34_4_debiased']) & \
                (sf_sample['t11_arms_number_a31_1_debiased'] >= sf_sample['t11_arms_number_a36_more_than_4_debiased']) & \
                (sf_sample['t11_arms_number_a31_1_debiased'] >= sf_sample['t11_arms_number_a37_cant_tell_debiased'])
    barred = (sf_sample['t03_bar_a06_bar_debiased'] >= 0.4)
    unbarred = (sf_sample['t03_bar_a06_bar_debiased'] < 0.4)

    axc = fig.add_subplot(111)    # The big subplot
    axc.spines['top'].set_color('none')
    axc.spines['bottom'].set_color('none')
    axc.spines['left'].set_color('none')
    axc.spines['right'].set_color('none')
    axc.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    axc.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',labelpad=10,fontsize=20)
    axc.set_ylabel('SFR '+r'$[\log\/M_\odot/\mathrm{yr}]$',labelpad=3,fontsize=20)

    # 1-armed spirals: barred and unbarred

    ax1 = fig.add_subplot(221)
    h = ax1.hist2d(mass,sfr,bins=50,cmap = cm.gray_r, norm=LogNorm())
    ax1.set_xlim(8,11.5)
    ax1.set_ylim(-3.9,2)
    ax1.get_xaxis().set_ticks([])
    onearm_nobar = sf_sample[spiral & an_1 & unbarred]
    ax1.scatter(onearm_nobar['MEDIAN_MASS'],onearm_nobar['MEDIAN_SFR'], s = 2, color='red', marker='o')
    plot_fits('SF galaxies',mass,sfr,None,ax1,'black',lw=1,ls='-')
    plot_fits('1-armed unbarred',onearm_nobar['MEDIAN_MASS'],onearm_nobar['MEDIAN_SFR'],None,ax1,'red')
    ax1.text(6.2,1.5,'1-armed, unbarred (%i)' % len(onearm_nobar),ha='left',fontsize=16)

    ax2 = fig.add_subplot(222)
    h = ax2.hist2d(mass,sfr,bins=50,cmap = cm.gray_r, norm=LogNorm())
    ax2.set_xlim(8,11.5)
    ax2.set_ylim(-3.9,2)
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    onearm_bar = sf_sample[spiral & an_1 & barred]
    ax2.scatter(onearm_bar['MEDIAN_MASS'],onearm_bar['MEDIAN_SFR'], s = 2, color='blue', marker='o')
    plot_fits('SF galaxies',mass,sfr,None,ax2,'black',lw=1,ls='-')
    plot_fits('1-armed unbarred',onearm_bar['MEDIAN_MASS'],onearm_bar['MEDIAN_SFR'],None,ax2,'blue')
    ax2.text(6.2,1.5,'1-armed, barred (%i)' % len(onearm_bar),ha='left',fontsize=16)

    # Merging galaxies: barred and unbarred

    with fits.open('%s/mergers/mergers_gz2.fits' % gzm_path) as f:
        data = f[1].data

    sf_mergers = (data['bpt'] == 1)
    barred_m = (data['t03_bar_a06_bar_debiased'] >= 0.4)
    unbarred_m = (data['t03_bar_a06_bar_debiased'] < 0.4)

    ax3 = fig.add_subplot(223)
    h = ax3.hist2d(mass,sfr,bins=50,cmap = cm.gray_r, norm=LogNorm())
    ax3.set_xlim(8,11.5)
    ax3.set_ylim(-3.9,2)
    merger_nobar = data[sf_mergers & unbarred_m]
    mb = ax3.scatter(merger_nobar['MEDIAN_MASS'],merger_nobar['MEDIAN_SFR'],c=merger_nobar['mass_ratio'], edgecolor='none',s = 50, marker='.', cmap=cm.RdBu, vmin=1.,vmax=10.)
    plot_fits('SF galaxies',mass,sfr,None,ax3,'black',lw=1,ls='-')
    plot_fits('Merging unbarred',merger_nobar['MEDIAN_MASS'],merger_nobar['MEDIAN_SFR'],None,ax3,'red',ls='-.')
    ax3.text(6.2,1.5,'Merging, unbarred (%i)' % len(merger_nobar),ha='left',fontsize=16)

    axcb1 = plt.axes([0.40, 0.18, 0.03, 0.12]) 
    cb1 = plt.colorbar(mb,cax = axcb1, orientation="vertical",ticks=[1,3,5,10])
    cb1.set_label('Mass ratio',fontsize=14)

    ax4 = fig.add_subplot(224)
    h = ax4.hist2d(mass,sfr,bins=50,cmap = cm.gray_r, norm=LogNorm())
    ax4.set_xlim(8,11.5)
    ax4.set_ylim(-3.9,2)
    ax4.get_yaxis().set_ticks([])
    merger_bar = data[sf_mergers & barred_m]
    ax4.scatter(merger_bar['MEDIAN_MASS'],merger_bar['MEDIAN_SFR'],c=merger_bar['mass_ratio'], edgecolor='none',s = 50, marker='.', cmap=cm.RdBu, vmin=1.,vmax=10.)
    plot_fits('SF galaxies',mass,sfr,None,ax4,'black',lw=1,ls='-')
    plot_fits('Merging unbarred',merger_bar['MEDIAN_MASS'],merger_bar['MEDIAN_SFR'],None,ax4,'blue',ls='-.')
    ax4.text(6.2,1.5,'Merging, barred (%i)' % len(merger_bar),ha='left',fontsize=16)

    axcb2 = plt.axes([0.80, 0.18, 0.03, 0.12]) 
    cb2 = plt.colorbar(mb,cax = axcb2, orientation="vertical",ticks=[1,3,5,10])
    cb2.set_label('Mass ratio',fontsize=14)

    box = axc.get_position()
    axColorbar = plt.axes([box.x1*1.02, box.y0, 0.01, box.height])
    cb = plt.colorbar(h[3],cax = axColorbar, orientation="vertical")
    cb.set_label(r'$N_\mathrm{star-forming\/galaxies}$' ,fontsize=16)

    fig.savefig('%s/tm_bar_trigger.pdf' % fig_path, dpi=200)

    return None

def plot_ms_bulge(sf_sample,weighted=False,contour=False,plurality=False,verbose=False):

    # Plot

    fig = plt.figure(12,(12,10))
    fig.clf()
    filestr=''

    fig.subplots_adjust(left=0.08,bottom=0.15,hspace=0,wspace=0)

    mass,sfr,sfr_err,ssfr,ssfr_err = get_mass_sfr(sf_sample)
    mass_bins,sfr_bins,ssfr_bins = bins()
    h,xedges,yedges = np.histogram2d(mass,sfr,bins=(mass_bins,sfr_bins))

    axc = fig.add_subplot(111)    # The big subplot
    axc.spines['top'].set_color('none')
    axc.spines['bottom'].set_color('none')
    axc.spines['left'].set_color('none')
    axc.spines['right'].set_color('none')
    axc.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    axc.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',labelpad=10,fontsize=20)
    axc.set_ylabel('SFR '+r'$[\log\/M_\odot/\mathrm{yr}]$',labelpad=3,fontsize=20)

    # Plot each morphological category

    responses = ('a10_no_bulge','a11_just_noticeable','a12_obvious','a13_dominant')
    bulge_tasks = ['t05_bulge_prominence_'+r for r in responses]
    colors = ('red','orange','yellow','green')

    for idx, (a,c) in enumerate(zip(bulge_tasks,colors)):

        bulge_label = a[25:]

        ax = fig.add_subplot(2,2,idx+1)
        h = ax.hist2d(mass,sfr,bins=50,cmap = cm.gray_r, norm=LogNorm())
        ax.set_ylim(-3,2)
        if idx in (0,1):
            ax.get_xaxis().set_ticks([])
        if idx in (1,3):
            ax.get_yaxis().set_ticks([])

        rcopy = list(bulge_tasks)
        thistask = rcopy.pop(idx)
        notedgeon = (sf_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sf_sample['t02_edgeon_a05_no_debiased'] > 0.715)
        morph_sample = sf_sample[notedgeon & (sf_sample['%s_flag' % a] == 1)] 
        pl = sf_sample[notedgeon & (sf_sample['%s_debiased' % a] >= sf_sample['%s_debiased' % rcopy[0]]) & (sf_sample['%s_debiased' % a] >= sf_sample['%s_debiased' % rcopy[1]]) & (sf_sample['%s_debiased' % a] >= sf_sample['%s_debiased' % rcopy[2]])] 

        # Two sets of plots: one weights histogram by debiased vote fraction per galaxy; other shows discrete categories from GZ2 flags.
        if weighted:
            filestr='_weighted'
            disks = sf_sample[notedgeon]
            if contour:
                hc,xc,yc = np.histogram2d(disks['MEDIAN_MASS'],disks['MEDIAN_SFR'],bins=(mass_bins,sfr_bins),weights=disks['%s_debiased' % a])
                levels=10**(np.linspace(0,3,12))
                CS = ax.contour(mass_bins[1:],sfr_bins[1:],hc.T,levels,colors=c)
                filestr += '_contour'
            else:
                h = ax.hist2d(disks['MEDIAN_MASS'],disks['MEDIAN_SFR'],bins=50,cmap = cm.RdYlGn, weights=disks['%s_debiased' % a],vmin=0.01,vmax=100.,norm=LogNorm())
            yval_weights = disks['%s_debiased' % a]
            plot_fits(bulge_label,disks['MEDIAN_MASS'],disks['MEDIAN_SFR'],yval_weights,ax,c,morph=a,verbose=verbose)
        else:
            filestr = ''
            if plurality:
                plotset = pl
                filestr += '_plurality'
            else:
                plotset = morph_sample

            if len(plotset) > 0:
                ax.scatter(plotset['MEDIAN_MASS'],plotset['MEDIAN_SFR'], s = 2, color=c, marker='o')
                try:
                    plot_fits(bulge_label,plotset['MEDIAN_MASS'],plotset['MEDIAN_SFR'],None,ax,c,morph=a,verbose=verbose)
                except LinAlgError:
                    print 'Insufficient number of points (%i) to fit %s data' % (len(plotset),bulge_label)
            else:
                print 'No data for %s' % bulge_label

        ax.text(6.2,1.3,bulge_label, color='k')

        # Plot the best linear fits

        plot_fits(bulge_label,mass,sfr,None,ax,'black',lw=1,ls='-')
        ax.set_xlim(8,11.5)

    # Set the colorbar and labels at the end

    box = axc.get_position()
    axColorbar = plt.axes([box.x1*1.02, box.y0, 0.01, box.height])
    cb = plt.colorbar(h[3],cax = axColorbar, orientation="vertical")
    cb.set_label(r'$N_\mathrm{star-forming\/galaxies}$' ,fontsize=16)

    fig.savefig('%s/ms_bulge%s.pdf' % (fig_path,filestr), dpi=200)

    return None

def inclination_reddening():

    # What is the distribution of inclination angles for the various morphologies?

    # Get data from CasJobs-matched file
    filename = '%s/mpajhu_gz2_axisratios.fits' % fits_path
    with fits.open(filename) as f:
        data = f[1].data

    # Star-forming galaxies only
    sf = (data['bpt'] == 1) | (data['bpt'] == 2)
    redshift = (data['REDSHIFT'] <= z_upperlim) & (data['REDSHIFT'] >= z_lowerlim)
    absmag = data['PETROMAG_MR'] < absmag_lim

    sf_sample = data[sf & redshift & absmag]

    # Set up plots

    fig = plt.figure(13,(12,6))
    fig.clf()

    fig.subplots_adjust(left=0.08,bottom=0.15,hspace=0,wspace=0)

    axc = fig.add_subplot(111)    # The big subplot
    axc.spines['top'].set_color('none')
    axc.spines['bottom'].set_color('none')
    axc.spines['left'].set_color('none')
    axc.spines['right'].set_color('none')
    axc.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    axc.set_xlabel('Inclination angle [deg] (deproj. r-band axis ratio, exp. fit)',labelpad=10,fontsize=20)
    axc.set_ylabel('Count',labelpad=10,fontsize=20)

    # Disks and bars
    notedgeon = (sf_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sf_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sf_sample['t02_edgeon_a05_no_weight'] >= 20)
    barred = sf_sample[notedgeon & (sf_sample['t03_bar_a06_bar_debiased'] >= 0.4)] 
    unbarred = sf_sample[notedgeon & (sf_sample['t03_bar_a06_bar_debiased'] < 0.4)] 

    # Spiral arms
    spiral = (sf_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sf_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sf_sample['t04_spiral_a08_spiral_weight'] >= 20) & (sf_sample['t04_spiral_a08_spiral_debiased'] > 0.619)
    #morph_sample = sf_sample[spiral & (sf_sample['%s_flag' % a] == 1)] 

    # Find mean,std for each sample

    print '\nexp A/B: \n'

    print 'All star-forming galaxies (N = %6i): mean = %.2f +- %.2f' % (len(sf_sample),np.mean(sf_sample['expAB_r']),np.std(sf_sample['expAB_r']))
    print 'All         disk galaxies (N = %6i): mean = %.2f +- %.2f' % (notedgeon.sum(),np.mean(sf_sample[notedgeon]['expAB_r']),np.std(sf_sample[notedgeon]['expAB_r']))
    print 'All       spiral galaxies (N = %6i): mean = %.2f +- %.2f' % (spiral.sum(),np.mean(sf_sample[spiral]['expAB_r']),np.std(sf_sample[spiral]['expAB_r']))

    print ''
    print 'Arms number: '

    nbins=15
    rad2deg = 180./np.pi

    ax1 = fig.add_subplot(131)
    ax1.get_xaxis().set_ticks(np.arange(10)*10.)
    ax1.set_title('Arms number')

    for idx,(an,anc) in enumerate(zip(an_tasks,an_colors)):

        rcopy = list(an_tasks)
        thistask = rcopy.pop(idx)
        pl = sf_sample[spiral & (sf_sample['%s_debiased' % an] >= sf_sample['%s_debiased' % rcopy[0]]) & (sf_sample['%s_debiased' % an] >= sf_sample['%s_debiased' % rcopy[1]]) & (sf_sample['%s_debiased' % an] >= sf_sample['%s_debiased' % rcopy[2]]) & (sf_sample['%s_debiased' % an] >= sf_sample['%s_debiased' % rcopy[3]]) & (sf_sample['%s_debiased' % an] >= sf_sample['%s_debiased' % rcopy[4]])] 

        # Check with KS test - does it deviate from the broader sample of spirals?
        d,p = ks_2samp(sf_sample[spiral]['expAB_R'],pl['expAB_r'])

        print '\t%12s arms (N = %6i): mean = %.2f +- %.2f; KS-prob from all spirals = %.6f' % (an[20:],len(pl),np.mean(pl['expAB_r']),np.std(pl['expAB_r']),p)

        histML(np.arccos(pl['expAB_r'])*rad2deg, bins=nbins, ax=ax1, histtype='stepfilled', color=anc,range=(0,90),alpha=0.4)

    histML(np.arccos(sf_sample[spiral]['expAB_r'])*rad2deg, bins=nbins, ax=ax1, histtype='step', color='k',range=(0,90),lw=3)

    print ''
    print 'Arms winding: '

    ax2 = fig.add_subplot(132)
    ax2.get_xaxis().set_ticks(np.arange(10)*10.)
    ax2.get_yaxis().set_ticks([])
    ax2.set_title('Arms winding')

    for idx,(aw,awc) in enumerate(zip(aw_tasks,aw_colors)):

        rcopy = list(aw_tasks)
        thistask = rcopy.pop(idx)
        pl = sf_sample[spiral & (sf_sample['%s_debiased' % aw] >= sf_sample['%s_debiased' % rcopy[0]]) & (sf_sample['%s_debiased' % aw] >= sf_sample['%s_debiased' % rcopy[1]])] 

        # Check with KS test - does it deviate from the broader sample of spirals?
        d,p = ks_2samp(sf_sample[spiral]['expAB_R'],pl['expAB_r'])

        print '\t%12s arms (N = %6i): mean = %.2f +- %.2f; KS-prob from all spirals = %.6f' % (aw[21:],len(pl),np.mean(pl['expAB_r']),np.std(pl['expAB_r']),p)

        histML(np.arccos(pl['expAB_r'])*rad2deg, bins=nbins, ax=ax2, histtype='stepfilled', color=awc, range=(0,90),alpha=0.4)

    histML(np.arccos(sf_sample[spiral]['expAB_r'])*rad2deg, bins=nbins, ax=ax2, histtype='step', color='k',range=(0,90),lw=3)

    print ''
    print 'Bars: '

    ax3 = fig.add_subplot(133)
    ax3.get_yaxis().set_ticks([])
    ax3.set_title('Bars')

    # Check with KS test - does it deviate from the broader sample of disks?
    db,pb = ks_2samp(sf_sample[notedgeon]['expAB_R'],barred['expAB_r'])
    du,pu = ks_2samp(sf_sample[notedgeon]['expAB_R'],unbarred['expAB_r'])

    histML(np.arccos(barred['expAB_r'])*rad2deg, bins=nbins, ax=ax3, histtype='stepfilled', color='blue', range=(0,90),alpha=0.4)
    histML(np.arccos(unbarred['expAB_r'])*rad2deg, bins=nbins, ax=ax3, histtype='stepfilled', color='red', range=(0,90),alpha=0.4)
    histML(np.arccos(sf_sample[notedgeon]['expAB_r'])*rad2deg, bins=nbins, ax=ax3, histtype='step', color='k',range=(0,90),lw=3)

    print 'Barred   (N = %6i): mean = %.2f +- %.2f; KS-prob from all not edge-on feature/disks = %.6f' % (len(barred),np.mean(barred['expAB_r']),np.std(barred['expAB_r']),pb)
    print 'Unbarred (N = %6i): mean = %.2f +- %.2f; KS-prob from all not edge-on feature/disks = %.6f' % (len(unbarred),np.mean(unbarred['expAB_r']),np.std(unbarred['expAB_r']),pu)

    for ax in (ax1,ax2,ax3):
        ax.set_ylim(0,2000)


    fig.savefig('%s/inclination_reddening.pdf' % fig_path, dpi=200)

    return None

def make_all_plots():

    data = get_data()
    sf_sample = get_sample(data,starforming=True)

    plot_ms_arms_number(sf_sample,weighted=False,contour=False,plot_ssfr=False)
    plot_ms_arms_number(sf_sample,weighted=True,contour=False,plot_ssfr=False,verbose=True)
    plot_ms_arms_number(sf_sample,weighted=True,contour=True,plot_ssfr=False)
    plot_ms_arms_number(sf_sample,weighted=False,contour=False,plot_ssfr=True)
    plot_ms_arms_number(sf_sample,weighted=True,contour=False,plot_ssfr=True)
    plot_ms_arms_number(sf_sample,weighted=True,contour=True,plot_ssfr=True)

    plot_ms_arms_winding(sf_sample,weighted=False,plot_ssfr=False)
    plot_ms_arms_winding(sf_sample,weighted=True,plot_ssfr=False,verbose=True)
    plot_ms_arms_winding(sf_sample,weighted=False,plot_ssfr=True)
    plot_ms_arms_winding(sf_sample,weighted=True,plot_ssfr=True)

    plot_ms_bars(sf_sample,contour=False,plot_ssfr=False,verbose=True)
    plot_ms_bars(sf_sample,contour=True,plot_ssfr=False)
    plot_ms_bars(sf_sample,contour=False,plot_ssfr=True)
    plot_ms_bars(sf_sample,contour=True,plot_ssfr=True)

    plot_ms_mergers_both(sf_sample,plot_ssfr=False,verbose=True)
    plot_ms_mergers_both(sf_sample,plot_ssfr=True)

    plot_ms_bulge(sf_sample,weighted=False,contour=False,plurality=False,verbose=True)
    plot_ms_bulge(sf_sample,weighted=True,contour=False,plurality=False)
    plot_ms_bulge(sf_sample,weighted=True,contour=True,plurality=False)
    plot_ms_bulge(sf_sample,weighted=False,contour=False,plurality=True)

    plot_ms_greenpeas(sf_sample)

    sigma_mstar(sf_sample)

    tm_bar_trigger(sf_sample)

    fracdev(sf_sample)
    fracdev_cdf(sf_sample)

    inclination_reddening()

    return None

def check_missing_galaxies():

    allgals = get_data()
    sf_sample = get_sample(allgals)
    vl_sample = get_sample(allgals,starforming=False)

    # Total sample

    notedgeon_vl = vl_sample[(vl_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (vl_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (vl_sample['t02_edgeon_a05_no_weight'] >= 20)]
    notedgeon_sf = sf_sample[(sf_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sf_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sf_sample['t02_edgeon_a05_no_weight'] >= 20)]
    print 'All galaxies: %6i in full, %6i in SF sample. Cut removes %6i (%.1f percent)' % (len(vl_sample),len(sf_sample),len(vl_sample) - len(sf_sample),(len(vl_sample)-len(sf_sample))/float(len(vl_sample))*100.)
    print 'All    disks: %6i in full, %6i in SF sample. Cut removes %6i (%.1f percent)' % (len(notedgeon_vl),len(notedgeon_sf),len(notedgeon_vl) - len(notedgeon_sf),(len(notedgeon_vl)-len(notedgeon_sf))/float(len(notedgeon_vl))*100.)

    print ''

    # Arms number

    responses = ('a31_1','a32_2','a33_3','a34_4','a36_more_than_4','a37_cant_tell')
    arm_tasks = tuple(['t11_arms_number_'+r for r in responses])

    for idx,a in enumerate(arm_tasks):
        s = []
        for samp in (vl_sample,sf_sample):

            # Count number of galaxies in each morphology

            spiral = ((samp['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (samp['t02_edgeon_a05_no_debiased'] > 0.715) & (samp['t04_spiral_a08_spiral_weight'] >= 20) & (samp['t04_spiral_a08_spiral_debiased'] > 0.619))

            rcopy = list(arm_tasks)
            thistask = rcopy.pop(idx)
            pl = samp[spiral & (samp['%s_debiased' % a] >= samp['%s_debiased' % rcopy[0]]) & (samp['%s_debiased' % a] >= samp['%s_debiased' % rcopy[1]]) & (samp['%s_debiased' % a] >= samp['%s_debiased' % rcopy[2]]) & (samp['%s_debiased' % a] >= samp['%s_debiased' % rcopy[3]]) & (samp['%s_debiased' % a] >= samp['%s_debiased' % rcopy[4]])] 

            s.append(len(pl))

        print 'Morph: %35s -- %6i in full, %6i in SF sample. Cut removes %6i (%.1f percent)' % (a,s[0],s[1],s[0] - s[1],(s[0]-s[1])/float(s[0])*100.)

    print ''

    # Arms winding

    responses = ('a28_tight','a29_medium','a30_loose')
    arm_tasks = tuple(['t10_arms_winding_'+r for r in responses])

    for idx,a in enumerate(arm_tasks):
        s = []
        for samp in (vl_sample,sf_sample):

            # Count number of galaxies in each morphology

            spiral = ((samp['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (samp['t02_edgeon_a05_no_debiased'] > 0.715) & (samp['t04_spiral_a08_spiral_weight'] >= 20) & (samp['t04_spiral_a08_spiral_debiased'] > 0.619))

            rcopy = list(arm_tasks)
            thistask = rcopy.pop(idx)
            pl = samp[spiral & (samp['%s_debiased' % a] >= samp['%s_debiased' % rcopy[0]]) & (samp['%s_debiased' % a] >= samp['%s_debiased' % rcopy[1]])] 

            s.append(len(pl))

        print 'Morph: %35s -- %6i in full, %6i in SF sample. Cut removes %6i (%.1f percent)' % (a,s[0],s[1],s[0] - s[1],(s[0]-s[1])/float(s[0])*100.)

    print ''

    # Bar

    b = []
    ub = []
    for samp in (vl_sample,sf_sample):

        # Count number of galaxies in each morphology

        notedgeon = (samp['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (samp['t02_edgeon_a05_no_debiased'] > 0.715) & (samp['t02_edgeon_a05_no_weight'] >= 20)
        barred = samp[notedgeon & (samp['t03_bar_a06_bar_debiased'] >= 0.4)] 
        unbarred = samp[notedgeon & (samp['t03_bar_a06_bar_debiased'] < 0.4)] 
        
        b.append(len(barred))
        ub.append(len(unbarred))

    print 'Morph:                              barred -- %6i in full, %6i in SF sample. Cut removes %6i (%.1f percent)' % (b[0],b[1],b[0] - b[1],(b[0]-b[1])/float(b[0])*100.)
    print 'Morph:                            unbarred -- %6i in full, %6i in SF sample. Cut removes %6i (%.1f percent)' % (ub[0],ub[1],ub[0] - ub[1],(ub[0]-ub[1])/float(ub[0])*100.)


    print ''


    return None

def volume_limited(data):

    # Find the largest volume-limited star-forming sample in MPA-JHU/GZ2

    redshift = data['REDSHIFT']
    appmag = data['PETROMAG_R']
    absmag = data['PETROMAG_MR']

    # What is the appropriate abs m     
    # appmag should be the detection limit of SDSS in r-band - 23.1 in survey, but 17.7 in GZ2 cutoff (using both Stripe 82 and Legacy DR7)ag limit as a function of redshift?

    appmag_lim = 17.7
    wmap9 = cosmology.WMAP9
    zarr = np.linspace(z_lowerlim,0.25,100)
    absmag_lim_arr = [(appmag_lim*u.mag)- mu for mu in wmap9.distmod(zarr)]
    ngals = [np.sum( (redshift <= z) & (absmag < m.value) & ((data['bpt'] == 1) | (data['bpt'] == 2)) ) for z,m in zip(zarr,absmag_lim_arr)]

    # Plot size of sample as function of limit

    fig = plt.figure(14,(12,6))
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(zarr,ngals)
    ax.set_xlim(zarr[0],zarr[-1])
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Number of GZ2 galaxies')

    # Add second axis for abs. mag
    ax2=ax.twiny()
    ax2.set_xlim(zarr[0],zarr[-1])
    new_tick_locations = np.linspace(zarr[0],zarr[-1],6)
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(['%.2f' % (appmag_lim - wmap9.distmod(z).value) for z in new_tick_locations])
    ax2.set_xlabel('Abs. mag limit')

    return None

def sfr_completeness(data):

    allgals = get_sample(data,starforming=False)
    sf = get_sample(data,starforming=True)
    mass_all,sfr_all,sfr_err_all,ssfr_all,ssfr_err_all = get_mass_sfr(allgals)
    mass_sf,sfr_sf,sfr_err_sf,ssfr_sf,ssfr_err_sf = get_mass_sfr(sf)

    # Plot SFR galaxies as grayscale contour
    mass_bins,sfr_bins,ssfr_bins = bins()

    fig = plt.figure(1,(10,6))
    fig.clf()
    ax = fig.add_subplot(111)
    h = ax.hist2d(mass_sf,sfr_sf,bins=50,cmap = cm.gray_r, norm=LogNorm())

    ax.scatter(mass_all,sfr_all)
    ax.set_xlim(8,11.5)

    return None

def timethis(tstart,description):
    tend = time.time()
    print 'Time elapsed for %s: %i seconds' % (description, tend - tstart)
    return tend

# Run plot-making routines

#make_all_plots()

