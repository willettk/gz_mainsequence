import numpy as np
import matplotlib

from matplotlib import rc
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.font_manager import FontProperties
from astropy.io import fits

plt.ion()

gz_path = '/Users/willettk/Astronomy/Research/GalaxyZoo'
ms_path = '%s/gzmainsequence' % gz_path
fig_path = '%s/gzmainsequence/figures' % gz_path
fits_path = '%s/fits' % gz_path

def get_data():

    filename = '%s/mpajhu_gz2.fits' % fits_path

    with fits.open(filename) as f:
        data = f[1].data

    return data

def get_sfr_sample(data):

    # Find starforming galaxies

    sf = (data['bpt'] == 1) #| (data['bpt'] == 2)
    redshift = data['REDSHIFT'] <= 0.1
    absmag = data['PETROMAG_MR'] < -19.5

    sfr_sample = data[sf & redshift]

    return sfr_sample

def get_mass_sfr(sfr_sample):

    mass_sfr_good = (sfr_sample['MEDIAN_MASS'] > -99) & (sfr_sample['MEDIAN_SFR'] > -99)

    mass = sfr_sample[mass_sfr_good]['MEDIAN_MASS']
    sfr = sfr_sample[mass_sfr_good]['MEDIAN_SFR']
    sfr16_err = np.abs(sfr_sample[mass_sfr_good]['P16_SFR'] - sfr_sample[mass_sfr_good]['MEDIAN_SFR'])
    sfr84_err = np.abs(sfr_sample[mass_sfr_good]['P84_SFR'] - sfr_sample[mass_sfr_good]['MEDIAN_SFR']) 
    sfr_err = (sfr16_err + sfr84_err) / 2.

    return mass,sfr,sfr_err

def bins(mass_min=6,mass_max=13,sfr_min=-5,sfr_max=2,nbins=50):

    mass_bins = np.linspace(mass_min,mass_max,nbins)
    sfr_bins = np.linspace(sfr_min,sfr_max,nbins)

    return mass_bins,sfr_bins
    
def fit_mass_sfr(sample,weighted=True):

    # Retrieve mass, star formation rate, and error for subsample from MPA-JHU catalog
    mass,sfr,sfr_err = get_mass_sfr(sample)

    # Use mean difference between 16th and 84th percentiles to the median value for the weights on fit

    w = 1./(sfr_err**2)
    weights = None if weighted else w

    # Find coefficients with polyfit

    (a,b),v = np.polyfit(mass,sfr,1,full=False,w=weights,cov=True)

    return a,b,v

def plot_fits(label,data,axis,color,lw=2,ls='--',legend=False,verbose=False,weighted=True):

    if weighted:
        prefix = ''
    else:
        prefix = 'un'

    a,b,v = fit_mass_sfr(data,weighted=weighted)
    
    xarr = np.linspace(6,12,100)
    line = axis.plot(xarr,np.polyval([a,b],xarr),color=color,linestyle=ls,linewidth=lw)

    '''
    if legend:
        axis.legend(['weighted','unweighted'],loc=4,shadow=True,fancybox=True,fontsize=10)
    '''

    if verbose:
        print 'Best fit (%10s) for %sweighted data: a,b = %.2f,%.2f' % (label,prefix,a,b)
        print 'Best fit (%10s) for %sweighted data: sigma_a,sigma_b=%.2e,%.2e' % (label,prefix,v[0,0],v[1,1])
        print ''


    return a,b

def plot_ms_arms_number(sfr_sample):

    # Plot

    fig = plt.figure(1,(10,6))
    fig.clf()

    fig.subplots_adjust(left=0.08,bottom=0.15,right=0.9,hspace=0,wspace=0)

    mass,sfr,sfr_err = get_mass_sfr(sfr_sample)
    mass_bins,sfr_bins = bins()
    h,xedges,yedges = np.histogram2d(mass,sfr,bins=(mass_bins,sfr_bins))

    #ax.scatter(mass,sfr,marker='o',color='cyan')
    #ax.contour(mass_bins[:-1],sfr_bins[:-1],h)

    # Plot each one

    arm_tasks = ('a31_1','a32_2','a33_3','a34_4','a36_more_than_4','a37_cant_tell')
    arm_label = ('1','2','3','4','5+','??')
    colors = ('red','orange','yellow','green','blue','purple')
    for idx, (a,c,al) in enumerate(zip(arm_tasks,colors,arm_label)):

        ax = fig.add_subplot(2,3,idx+1)
        h = ax.hist2d(mass,sfr,bins=50,cmap = cm.gray_r, norm=LogNorm())
        ax.set_xlim(6,11.5)
        ax.set_ylim(-3.9,2)

        if idx == 4:
            ax.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',fontsize=16)
        if idx < 3:
            ax.get_xaxis().set_ticks([])
        if idx == 0 or idx == 3:
            ax.set_ylabel('SFR '+r'$[\log\/M_\odot/\mathrm{yr}]$',fontsize=16)
        else:
            ax.get_yaxis().set_ticks([])


        spiral = (sfr_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sfr_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sfr_sample['t04_spiral_a08_spiral_weight'] >= 20) & (sfr_sample['t04_spiral_a08_spiral_debiased'] > 0.619)
        n1 = sfr_sample[spiral & (sfr_sample['t11_arms_number_%s_flag' % a] == 1)] 
        ax.scatter(n1['MEDIAN_MASS'],n1['MEDIAN_SFR'], s = 2, color=c, marker='o')

        ax.text(6.2,1.0,r'$N_{arms} = $%s' % al, color='k',fontsize=18)

        # Plot the linear fits
        plot_fits(al,n1,ax,c)
        plot_fits('SF galaxies',sfr_sample,ax,'black',lw=1,ls='-')

        '''
        # Print number of galaxies in each category

        print '%i %s galaxies' % (len(n1),a)
        '''


    box = ax.get_position()
    axColorbar = plt.axes([box.x0*1.05 + box.width * 0.95, box.y0, 0.01, box.height*2])
    cb = plt.colorbar(h[3],cax = axColorbar, orientation="vertical")
    cb.set_label(r'$N_\mathrm{star-forming\/galaxies}$' ,fontsize=16)

    fig.savefig('%s/ms_arms_number.pdf' % fig_path, dpi=200)

    return None

def plot_ms_bars(sfr_sample,contour=False):

    # Plot

    fig = plt.figure(2,(11,6))
    fig.clf()
    fig.subplots_adjust(left=0.08,bottom=0.15,right=0.9,hspace=0,wspace=0)
    filestr=''

    mass,sfr,sfr_err = get_mass_sfr(sfr_sample)
    mass_bins,sfr_bins = bins()
    h,xedges,yedges = np.histogram2d(mass,sfr,bins=(mass_bins,sfr_bins))

    notedgeon = (sfr_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sfr_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sfr_sample['t02_edgeon_a05_no_weight'] >= 20)

    barred = sfr_sample[notedgeon & (sfr_sample['t03_bar_a06_bar_debiased'] >= 0.4)] 
    unbarred = sfr_sample[notedgeon & (sfr_sample['t03_bar_a06_bar_debiased'] < 0.4)] 

    bar_data = (barred,unbarred)
    bar_text = ('Barred','Unbarred')
    color = ('blue','red')

    # Plot barred and unbarred
    for idx, (b,c,t) in enumerate(zip(bar_data,color,bar_text)):

        ax = fig.add_subplot(1,2,idx+1)
        h2 = ax.hist2d(mass,sfr,bins=50,cmap = cm.gray_r, norm=LogNorm())
        ax.set_xlim(6,11.5)
        ax.set_ylim(-4,2)
        ax.set_xlabel('Stellar mass (log '+r'$M/M_\odot$)',fontsize=20)

        if idx == 0:
            ax.set_ylabel('SFR '+r'$[\log\/M_\odot/\mathrm{yr}]$',fontsize=20)
        else:
            ax.get_yaxis().set_ticks([])

        # Try a contour plot??

        if contour:
            hb,xb,yb = np.histogram2d(b['MEDIAN_MASS'],b['MEDIAN_SFR'],bins=(mass_bins,sfr_bins))
            levels=10**(np.linspace(0,2,8))
            CS = ax.contour(mass_bins[1:],sfr_bins[1:],hb.T,levels,colors=c)
            filestr='_contour'
        else:
            ax.scatter(b['MEDIAN_MASS'],b['MEDIAN_SFR'], s=2, color=c, marker='o')

        ax.text(6.2,1.4,t,color=c,fontsize=18)

        # Plot the best linear fit

        plot_fits(t,b,ax,c)
        plot_fits(t,sfr_sample,ax,'black',lw=1,ls='-')

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
    
def plot_ms_arms_winding(sfr_sample,weight_by_pmed=False):

    # Plot

    fig = plt.figure(3,(10,4))
    fig.clf()
    filestr=''

    fig.subplots_adjust(left=0.08,bottom=0.15,hspace=0,wspace=0)

    mass,sfr,sfr_err = get_mass_sfr(sfr_sample)
    mass_bins,sfr_bins = bins()
    h,xedges,yedges = np.histogram2d(mass,sfr,bins=(mass_bins,sfr_bins))

    # Plot each morphological category

    arm_tasks = ('a28_tight','a29_medium','a30_loose')
    colors = ('red','green','blue')
    for idx, (a,c) in enumerate(zip(arm_tasks,colors)):

        ax = fig.add_subplot(1,3,idx+1)
        h = ax.hist2d(mass,sfr,bins=50,cmap = cm.gray_r, norm=LogNorm())
        ax.set_xlim(6,11.5)
        ax.set_ylim(-4,2)
        if idx == 0:
            ax.set_ylabel('SFR '+r'$[\log\/M_\odot/\mathrm{yr}]$',fontsize=16)
        if idx == 1:
            ax.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',fontsize=16)

        if idx > 0:
            ax.get_yaxis().set_ticks([])

        spiral = (sfr_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sfr_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sfr_sample['t04_spiral_a08_spiral_weight'] >= 20) & (sfr_sample['t04_spiral_a08_spiral_debiased'] > 0.619)
        n1 = sfr_sample[spiral & (sfr_sample['t10_arms_winding_%s_flag' % a] == 1)] 

        # Two sets of plots: one weights histogram by debiased vote fraction per galaxy; other shows discrete categories from GZ2 flags.
        if weight_by_pmed:
            spirals = sfr_sample[spiral]
            h = ax.hist2d(spirals['MEDIAN_MASS'],spirals['MEDIAN_SFR'],bins=50,cmap = cm.RdYlGn, weights=spirals['t10_arms_winding_%s_debiased' % a],vmin=0.01,vmax=100.,norm=LogNorm())
            filestr='_weighted_pmed'
            cb_label = r'$w_\mathrm{\phi}$'
        else:
            ax.scatter(n1['MEDIAN_MASS'],n1['MEDIAN_SFR'], s = 2, color=c, marker='o')
            cb_label = r'$N_\mathrm{star-forming\/galaxies}$'

        arm_label = a[4:]
        ax.text(6.2,1.3,r'$\phi_{arms} = $%s' % arm_label, color='k')

        # Plot the best linear fits

        plot_fits(arm_label,n1,ax,c)
        plot_fits(arm_label,sfr_sample,ax,'black',lw=1,ls='-')

        '''
        # Print number of galaxies in each category

        print '%i %s galaxies' % (len(n1),a)
        '''

    # Set the colorbar and labels at the end

    box = ax.get_position()
    axColorbar = plt.axes([box.x0*1.05 + box.width * 0.93, box.y0, 0.01, box.height])
    cb = plt.colorbar(h[3],cax = axColorbar, orientation="vertical")
    cb.set_label(cb_label,fontsize=16)

    fig.savefig('%s/ms_arms_winding%s.pdf' % (fig_path,filestr), dpi=200)

    return None

def plot_ms_mergers(sfr_sample):

    # Plot

    fig = plt.figure(4,(10,8))
    fig.clf()

    fig.subplots_adjust(left=0.08,bottom=0.15,hspace=0,wspace=0)

    mass,sfr,sfr_err = get_mass_sfr(sfr_sample)
    mass_bins,sfr_bins = bins()
    h,xedges,yedges = np.histogram2d(mass,sfr,bins=(mass_bins,sfr_bins))

    # Plot star-forming galaxies

    ax = fig.add_subplot(111)
    h = ax.hist2d(mass,sfr,bins=50,cmap = cm.gray_r, norm=LogNorm())
    ax.set_xlim(6,11.5)
    ax.set_ylim(-4,2)
    ax.set_ylabel('log SFR '+r'$[M_\odot/\mathrm{yr}]$',fontsize=16)
    ax.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',fontsize=16)

    # Plot mergers data

    with fits.open('%s/mergers/mergers_mpajhu_bpt.fits' % ms_path) as f:
        data = f[1].data

    #sf_mergers = data[(data['bpt'] == 1) & (data['mass_ratio'] <= 3)]
    sf_mergers = data[(data['bpt'] == 1)]

    sc = ax.scatter(sf_mergers['MEDIAN_MASS'],sf_mergers['MEDIAN_SFR'], c=sf_mergers['mass_ratio'], edgecolor='none',s = 50, marker='.', cmap=matplotlib.cm.RdBu, vmin=1.,vmax=10.)

    ax.text(6.2,1.3,'Mergers', color='black',fontsize=20)

    # Plot the best linear fits

    #au,bu = plot_fits('Mergers',sf_mergers,ax,'red')
    a1,a0 = plot_fits('Star-forming galaxies',sfr_sample,ax,'black',lw=1,ls='-')

    # How many mergers fall above and below?

    diff = sf_mergers['MEDIAN_SFR'] - (a0 + a1*sf_mergers['MEDIAN_MASS'])

    # Fix same slope but different offset. What's the average difference between SF MS and mergers?

    mass,sfr,sfr_err = get_mass_sfr(sf_mergers)

    rms = []
    xarr = np.linspace(0,1,100)
    for x in xarr:
        rms.append(np.sqrt(np.sum((sfr - (a0 + x + a1*mass))**2)))

    offset = xarr[(np.abs(rms)).argmin()]
    uline = ax.plot(np.linspace(6,12,100),np.polyval([a1,a0+offset],np.linspace(6,12,100)),color='red',linestyle='--',linewidth=1)

    ax.text(6.2,0.9,'Offset = %.3f dex' % offset, color='black',fontsize=15)

    # Set the colorbars and labels

    box = ax.get_position()
    axColorbar = plt.axes([box.x0 + box.width * 1.02, box.y0, 0.01, box.height])
    cb = plt.colorbar(h[3],cax = axColorbar, orientation="vertical")
    cb.set_label(r'$N_\mathrm{star-forming\/galaxies}$' ,fontsize=16)

    axcb2 = plt.axes([0.75, 0.20, 0.03, 0.25]) 
    cb2 = plt.colorbar(sc,cax = axcb2, orientation="vertical",ticks=[1,3,5,10])
    cb2.set_label('Merger ratio',fontsize=14)

    fig.savefig('%s/ms_mergers.pdf' % fig_path, dpi=200)

    return None

def plot_ms_greenpeas(sfr_sample):

    # Plot

    fig = plt.figure(5,(10,8))
    fig.clf()

    fig.subplots_adjust(left=0.08,bottom=0.15,hspace=0,wspace=0)

    mass,sfr,sfr_err = get_mass_sfr(sfr_sample)
    mass_bins,sfr_bins = bins()
    h,xedges,yedges = np.histogram2d(mass,sfr,bins=(mass_bins,sfr_bins))

    # Plot star-forming galaxies

    ax = fig.add_subplot(111)
    h = ax.hist2d(mass,sfr,bins=50,cmap = cm.gray_r, norm=LogNorm())
    ax.set_xlim(6,11.5)
    ax.set_ylim(-4,2)
    ax.set_ylabel('log SFR '+r'$[M_\odot/\mathrm{yr}]$',fontsize=16)
    ax.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',fontsize=16)

    # Plot green peas

    with fits.open('%s/greenpeas.fits' % ms_path) as f:
        data = f[1].data

    sc = ax.scatter(data['M_STELLAR'],np.log10(data['SFR']), color='green',s = 10, marker='o')

    ax.text(6.2,1.3,'Green peas', color='green')

    # Plot the best linear fits

    #au,bu = plot_fits('Mergers',sf_mergers,ax,'red')
    a1,a0 = plot_fits('Star-forming galaxies',sfr_sample,ax,'black',lw=1,ls='-')

    # Set the colorbars and labels

    box = ax.get_position()
    axColorbar = plt.axes([box.x0 + box.width * 1.02, box.y0, 0.01, box.height])
    cb = plt.colorbar(h[3],cax = axColorbar, orientation="vertical")
    cb.set_label(r'$N_\mathrm{star-forming\/galaxies}$' ,fontsize=16)

    fig.savefig('%s/ms_greenpeas.pdf' % fig_path, dpi=200)

    return None

def sigma_mstar(sfr_sample):

    '''
    Test whether populations are likely to be consistent by plotting dispersion as function of M_star for bins of M_star
    Overplot for different populations
    '''

    mass,sfr,sfr_err = get_mass_sfr(sfr_sample)
    mass_bins,sfr_bins = bins(mass_max=11.5,nbins=15)
    deltamass = mass_bins[1] - mass_bins[0]

    fig = plt.figure(6,(10,8))
    fig.clf()

    # Bars
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

    notedgeon = (sfr_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sfr_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sfr_sample['t02_edgeon_a05_no_weight'] >= 20)
    barred = sfr_sample[notedgeon & (sfr_sample['t03_bar_a06_bar_debiased'] >= 0.4)] 
    unbarred = sfr_sample[notedgeon & (sfr_sample['t03_bar_a06_bar_debiased'] < 0.4)] 

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
    ax.set_xlim(6,11.5)
    ax.set_ylim(0,1.2)
    ax.set_ylabel(r'$\sigma_{SFR}~[\log M_\odot/\mathrm{yr}]$',fontsize=16)
    ax.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',fontsize=16)
    ax.set_title('Bar/no bar')

    # Arm number
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

    arm_tasks = ('a31_1','a32_2','a33_3','a34_4','a36_more_than_4','a37_cant_tell')
    arm_label = ('1','2','3','4','5+','??')
    colors = ('red','orange','yellow','green','blue','purple')
    for idx, (a,c,al) in enumerate(zip(arm_tasks,colors,arm_label)):

        spiral = (sfr_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sfr_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sfr_sample['t04_spiral_a08_spiral_weight'] >= 20) & (sfr_sample['t04_spiral_a08_spiral_debiased'] > 0.619)
        n1 = sfr_sample[spiral & (sfr_sample['t11_arms_number_%s_flag' % a] == 1)] 

        sigma_sfr,sigma_sfr_err,ngals = [],[],[]
        for mb in mass_bins:
            idx = (n1['MEDIAN_MASS'] > (mb-deltamass)) & (n1['MEDIAN_MASS'] <= (mb+deltamass))
            sigma_sfr.append(np.std(n1['MEDIAN_SFR'][idx]))
            try:
                sigma_sfr_err.append(1./np.sum(idx))
            except ZeroDivisionError:
                sigma_sfr_err.append(0.)
            ngals.append(np.sum(idx))

        gtr10 = (np.array(ngals) >= 10)
        ax.errorbar(mass_bins[gtr10],np.array(sigma_sfr)[gtr10],yerr=np.zeros_like(sigma_sfr)[gtr10],color=c,capsize=0)

    # Final plot labels and ranges
    ax.set_xlim(6,11.5)
    ax.set_ylim(0,1.2)
    ax.set_ylabel(r'$\sigma_{SFR}~[\log M_\odot/\mathrm{yr}]$',fontsize=16)
    ax.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',fontsize=16)
    ax.set_title('Arms number')

    # Arm winding
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

    arm_tasks = ('a28_tight','a29_medium','a30_loose')
    colors = ('red','green','blue')
    for idx, (a,c) in enumerate(zip(arm_tasks,colors)):

        spiral = (sfr_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sfr_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sfr_sample['t04_spiral_a08_spiral_weight'] >= 20) & (sfr_sample['t04_spiral_a08_spiral_debiased'] > 0.619)
        n1 = sfr_sample[spiral & (sfr_sample['t10_arms_winding_%s_flag' % a] == 1)] 

        sigma_sfr,sigma_sfr_err,ngals = [],[],[]
        for mb in mass_bins:
            idx = (n1['MEDIAN_MASS'] > (mb-deltamass)) & (n1['MEDIAN_MASS'] <= (mb+deltamass))
            sigma_sfr.append(np.std(n1['MEDIAN_SFR'][idx]))
            try:
                sigma_sfr_err.append(1./np.sum(idx))
            except ZeroDivisionError:
                sigma_sfr_err.append(0.)
            ngals.append(np.sum(idx))

        gtr10 = (np.array(ngals) >= 10)
        ax.errorbar(mass_bins[gtr10],np.array(sigma_sfr)[gtr10],yerr=np.array(sigma_sfr_err)[gtr10],color=c,capsize=0)

    # Final plot labels and ranges
    ax.set_xlim(6,11.5)
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

    with fits.open('%s/mergers/mergers_mpajhu_bpt.fits' % ms_path) as f:
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
    ax.set_xlim(6,11.5)
    ax.set_ylim(0,1.2)
    ax.set_ylabel(r'$\sigma_{SFR}~[\log M_\odot/\mathrm{yr}]$',fontsize=16)
    ax.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',fontsize=16)
    ax.set_title('Mergers')

    fig.set_tight_layout(True)
    fig.savefig('%s/sigma_mstar.pdf' % fig_path, dpi=200)

    return None

def plot_ms_merger_fraction(sfr_sample):

    # Plot

    fig = plt.figure(7,(10,8))
    fig.clf()

    fig.subplots_adjust(left=0.08,bottom=0.15,right=0.9,hspace=0,wspace=0)

    mass,sfr,sfr_err = get_mass_sfr(sfr_sample)
    mass_bins,sfr_bins = bins(nbins=30)
    h,xedges,yedges = np.histogram2d(mass,sfr,bins=(mass_bins,sfr_bins))

    # Plot star-forming galaxies

    ax = fig.add_subplot(111)
    levels=10**(np.linspace(0,5,15))
    CS = ax.contour(mass_bins[1:],sfr_bins[1:],h.T,levels,colors='black')
    ax.set_xlim(6,11.5)
    ax.set_ylim(-4,2)
    ax.set_ylabel('log SFR '+r'$[M_\odot/\mathrm{yr}]$',fontsize=16)
    ax.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',fontsize=16)

    # Plot mergers data

    with fits.open('%s/mergers/mergers_mpajhu_bpt.fits' % ms_path) as f:
        data = f[1].data

    #sf_mergers = data[(data['bpt'] == 1) & (data['mass_ratio'] <= 3)]
    sf_mergers = data[(data['bpt'] == 1)]

    #sc = ax.scatter(sf_mergers['MEDIAN_MASS'],sf_mergers['MEDIAN_SFR'], c=sf_mergers['mass_ratio'], edgecolor='none',s = 50, marker='.', cmap=matplotlib.cm.RdBu, vmin=1.,vmax=10.)
    hm,xm,ym = np.histogram2d(sf_mergers['MEDIAN_MASS'],sf_mergers['MEDIAN_SFR'],bins=(mass_bins,sfr_bins))
    mf = hm / h
    im = ax.imshow(mf.T, interpolation='nearest', origin='lower',extent=(mass_bins[0],mass_bins[-1],sfr_bins[0],sfr_bins[-1]),vmin=0.,vmax=1.)
    cb = plt.colorbar(im)

    ax.text(6.2,1.3,'Merger fraction', color='black',fontsize=20)

    # Plot the best linear fits

    #au,bu = plot_fits('Mergers',sf_mergers,ax,'red')
    a1,a0 = plot_fits('Star-forming galaxies',sfr_sample,ax,'black',lw=3,ls='-')

    # How many mergers fall above and below?

    diff = sf_mergers['MEDIAN_SFR'] - (a0 + a1*sf_mergers['MEDIAN_MASS'])

    # Fix same slope but different offset. What's the average difference between SF MS and mergers?

    mass,sfr,sfr_err = get_mass_sfr(sf_mergers)

    rms = []
    xarr = np.linspace(0,1,100)
    for x in xarr:
        rms.append(np.sqrt(np.sum((sfr - (a0 + x + a1*mass))**2)))

    offset = xarr[(np.abs(rms)).argmin()]
    uline = ax.plot(np.linspace(6,12,100),np.polyval([a1,a0+offset],np.linspace(6,12,100)),color='red',linestyle='--',linewidth=1)

    ax.text(6.2,0.9,'Offset = %.3f dex' % offset, color='black',fontsize=15)

    fig.savefig('%s/ms_merger_fraction.pdf' % fig_path, dpi=200)

    return None

def plot_ms_mergers_both(sfr_sample):

    #---------------------------------
    # Merger counts
    #---------------------------------

    fig = plt.figure(8,(14,7))
    fig.clf()

    #fig.subplots_adjust(left=0.08,bottom=0.15,right=0.90,hspace=0,wspace=0.05)

    mass,sfr,sfr_err = get_mass_sfr(sfr_sample)

    # Plot star-forming galaxies

    ax1 = fig.add_axes([0.07,0.10,0.42,0.85])
    h = ax1.hist2d(mass,sfr,bins=50,cmap = cm.gray_r, norm=LogNorm())
    ax1.set_xlim(6,11.5)
    ax1.set_ylim(-4,2)
    ax1.set_ylabel('log SFR '+r'$[M_\odot/\mathrm{yr}]$',fontsize=16)
    ax1.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',fontsize=16)

    # Plot mergers data

    with fits.open('%s/mergers/mergers_mpajhu_bpt.fits' % ms_path) as f:
        data = f[1].data

    sf_mergers = data[(data['bpt'] == 1)]

    sc = ax1.scatter(sf_mergers['MEDIAN_MASS'],sf_mergers['MEDIAN_SFR'], c=sf_mergers['mass_ratio'], edgecolor='none',s = 50, marker='.', cmap=matplotlib.cm.RdBu, vmin=1.,vmax=10.)

    ax1.text(6.2,1.3,'Mergers', color='black',fontsize=20)

    # Plot the best linear fits

    a1,a0 = plot_fits('Star-forming galaxies',sfr_sample,ax1,'black',lw=3,ls='-')

    # Fix same slope but different offset. 

    mass_mergers,sfr_mergers,sfr_err_mergers = get_mass_sfr(sf_mergers)

    rms = []
    xarr = np.linspace(0,1,100)
    for x in xarr:
        rms.append(np.sqrt(np.sum((sfr_mergers - (a0 + x + a1*mass_mergers))**2)))

    offset = xarr[(np.abs(rms)).argmin()]
    uline = ax1.plot(np.linspace(6,12,100),np.polyval([a1,a0+offset],np.linspace(6,12,100)),color='red',linestyle='--',linewidth=3)

    ax1.text(6.2,0.9,'Offset = %.3f dex' % offset, color='black',fontsize=15)

    # Set the colorbars and labels

    axcb1 = plt.axes([0.40, 0.20, 0.03, 0.25]) 
    cb1 = plt.colorbar(sc,cax = axcb1, orientation="vertical",ticks=[1,3,5,10])
    cb1.set_label('Mass ratio',fontsize=14)

    ax1.xaxis.labelpad = 20
    #---------------------------------
    # Merger fraction
    #---------------------------------

    mass_bins,sfr_bins = bins(nbins=30)
    h,xedges,yedges = np.histogram2d(mass,sfr,bins=(mass_bins,sfr_bins))

    # Plot star-forming galaxies

    ax2 = fig.add_axes([0.460,0.10,0.45,0.85])
    levels=10**(np.linspace(0,5,15))
    CS = ax2.contour(mass_bins[1:],sfr_bins[1:],h.T,levels,colors='black')
    ax2.set_xlim(6,11.5)
    ax2.set_ylim(-4,2)
    ax2.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',fontsize=16)

    hm,xm,ym = np.histogram2d(sf_mergers['MEDIAN_MASS'],sf_mergers['MEDIAN_SFR'],bins=(mass_bins,sfr_bins))
    mf = hm / h
    im = ax2.imshow(mf.T, interpolation='nearest', origin='lower',extent=(mass_bins[0],mass_bins[-1],sfr_bins[0],sfr_bins[-1]),vmin=0.,vmax=1.)

    axcb2 = plt.axes([0.90, 0.10, 0.02, 0.85]) 
    cb2 = plt.colorbar(im,cax = axcb2, orientation="vertical")
    cb2.set_label('Merger fraction',fontsize=16)

    ax2.text(6.2,1.3,'Merger fraction', color='black',fontsize=20)

    ax2.set_yticklabels([])

    # Plot the best linear fits

    a1,a0 = plot_fits('Star-forming galaxies',sfr_sample,ax2,'black',lw=3,ls='-')
    uline = ax2.plot(np.linspace(6,12,100),np.polyval([a1,a0+offset],np.linspace(6,12,100)),color='red',linestyle='--',linewidth=3)

    ax2.xaxis.labelpad = 20

    #fig.set_tight_layout(True)
    fig.savefig('%s/ms_mergers_both.pdf' % fig_path, dpi=200)

    return None
