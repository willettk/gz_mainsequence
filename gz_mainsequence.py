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

def get_data(mergers=False):

    if mergers:
        filename = '%s/mergers/mergers_mpajhu_gz2.fits' % ms_path
    else:
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

def bins():

    mass_bins = np.linspace(6,13,50)
    sfr_bins = np.linspace(-5,2,50)

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

def plot_fits(label,data,axis,color,lw=2,ls='--',legend=True,verbose=False,weighted=True):

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
            ax.set_ylabel('SFR '+r'$[M_\odot/\mathrm{yr}]$',fontsize=16)
        else:
            ax.get_yaxis().set_ticks([])


        spiral = (sfr_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sfr_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sfr_sample['t04_spiral_a08_spiral_weight'] >= 20) & (sfr_sample['t04_spiral_a08_spiral_debiased'] > 0.619)
        n1 = sfr_sample[spiral & (sfr_sample['t11_arms_number_%s_flag' % a] == 1)] 
        ax.scatter(n1['MEDIAN_MASS'],n1['MEDIAN_SFR'], s = 2, color=c, marker='o')

        ax.text(6.2,1.0,r'$N_{arms} = $%s' % al, color='k',fontsize=18)

        # Plot the linear fits
        plot_fits(al,n1,ax,c,legend=False,verbose=True,weighted=True)
        plot_fits('SF galaxies',sfr_sample,ax,'black',lw=1,ls='-',legend=False,weighted=True)

    box = ax.get_position()
    axColorbar = plt.axes([box.x0*1.05 + box.width * 0.95, box.y0, 0.01, box.height*2])
    cb = plt.colorbar(h[3],cax = axColorbar, orientation="vertical")
    cb.set_label(r'$N_\mathrm{star-forming\/galaxies}$' ,fontsize=16)

    fig.savefig('%s/ms_arms_number.pdf' % fig_path, dpi=200)

    return None

def plot_ms_bars(sfr_sample):

    # Plot

    fig = plt.figure(2,(11,6))
    fig.clf()
    fig.subplots_adjust(left=0.08,bottom=0.15,right=0.9,hspace=0,wspace=0)

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
        h = ax.hist2d(mass,sfr,bins=50,cmap = cm.gray_r, norm=LogNorm())
        ax.set_xlim(6,11.5)
        ax.set_ylim(-4,2)
        ax.set_xlabel('Stellar mass (log'+r'$\/M/M_\odot$)',fontsize=20)

        if idx == 0:
            ax.set_ylabel('SFR '+r'$[M_\odot/\mathrm{yr}]$',fontsize=16)
        else:
            ax.get_yaxis().set_ticks([])

        ax.scatter(b['MEDIAN_MASS'],b['MEDIAN_SFR'], s=2, color=c, marker='o')

        ax.text(6.2,1.4,t,color=c,fontsize=18)

        # Plot the best linear fit

        plot_fits(t,b,ax,c,legend=False)
        plot_fits(t,sfr_sample,ax,'black',lw=1,legend=False)


    box = ax.get_position()
    axColorbar = plt.axes([box.x0*1.05 + box.width * 0.95, box.y0, 0.01, box.height])
    cb = plt.colorbar(h[3],cax = axColorbar, orientation="vertical")
    cb.set_label(r'$N_\mathrm{star-forming\/galaxies}$' ,fontsize=16)

    fig.savefig('%s/ms_bar.pdf' % fig_path, dpi=200)

    return None
    
def plot_ms_arms_winding(sfr_sample):

    # Plot

    fig = plt.figure(3,(10,4))
    fig.clf()

    fig.subplots_adjust(left=0.08,bottom=0.15,hspace=0,wspace=0)

    mass,sfr,sfr_err = get_mass_sfr(sfr_sample)
    mass_bins,sfr_bins = bins()
    h,xedges,yedges = np.histogram2d(mass,sfr,bins=(mass_bins,sfr_bins))

    #ax.scatter(mass,sfr,marker='o',color='cyan')
    #ax.contour(mass_bins[:-1],sfr_bins[:-1],h)

    # Plot each one

    arm_tasks = ('a28_tight','a29_medium','a30_loose')
    colors = ('red','green','blue')
    for idx, (a,c) in enumerate(zip(arm_tasks,colors)):

        ax = fig.add_subplot(1,3,idx+1)
        h = ax.hist2d(mass,sfr,bins=50,cmap = cm.gray_r, norm=LogNorm())
        ax.set_xlim(6,11.5)
        ax.set_ylim(-4,2)
        if idx == 0:
            ax.set_ylabel('SFR '+r'$[M_\odot/\mathrm{yr}]$',fontsize=16)
        if idx == 1:
            ax.set_xlabel('Stellar mass [log '+r'$\/M/M_\odot$]',fontsize=16)
        #fig.colorbar(h[3], ax=ax)

        if idx > 0:
            ax.get_yaxis().set_ticks([])

        spiral = (sfr_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sfr_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sfr_sample['t04_spiral_a08_spiral_weight'] >= 20) & (sfr_sample['t04_spiral_a08_spiral_debiased'] > 0.619)
        n1 = sfr_sample[spiral & (sfr_sample['t10_arms_winding_%s_flag' % a] == 1)] 
        ax.scatter(n1['MEDIAN_MASS'],n1['MEDIAN_SFR'], s = 2, color=c, marker='o')

        arm_label = a[4:]
        ax.text(6.2,1.3,r'$\phi_{arms} = $%s' % arm_label, color='k')

        # Plot the best linear fits

        plot_fits(arm_label,n1,ax,c,legend=False)
        plot_fits(arm_label,sfr_sample,ax,'black',lw=1,legend=False)

    # Set the colorbar and labels at the end

    box = ax.get_position()
    axColorbar = plt.axes([box.x0*1.05 + box.width * 0.95, box.y0, 0.01, box.height])
    cb = plt.colorbar(h[3],cax = axColorbar, orientation="vertical")
    cb.set_label(r'$N_\mathrm{star-forming\/galaxies}$' ,fontsize=16)

    fig.savefig('%s/ms_arms_winding.pdf' % fig_path, dpi=200)

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

    ax.text(6.2,1.3,'Mergers', color='black')

    # Plot the best linear fits

    #au,bu = plot_fits('Mergers',sf_mergers,ax,'red',lw=2,legend=False)
    a1,a0 = plot_fits('Star-forming galaxies',sfr_sample,ax,'black',lw=1,legend=False)

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

def plot_ms_merger_fraction(sfr_sample):

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

    ax.text(6.2,1.3,'Mergers', color='black')

    # Plot the best linear fits

    #au,bu = plot_fits('Mergers',sf_mergers,ax,'red',lw=2,legend=False)
    a1,a0 = plot_fits('Star-forming galaxies',sfr_sample,ax,'black',lw=1,legend=False)

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

