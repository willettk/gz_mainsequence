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
ms_path = '%s/gz_mainsequence' % gz_path
fits_path = '%s/fits' % gz_path


def get_data():

    with fits.open('%s/mpajhu_gz2.fits' % fits_path) as f:
        data = f[1].data

    return data

def subsample(data):

    # Find starforming galaxies

    sf = (data['bpt'] == 1) #| (data['bpt'] == 2)
    redshift = data['REDSHIFT'] <= 0.1
    absmag = data['PETROMAG_MR'] < -19.5

    spiral = (data['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (data['t02_edgeon_a05_no_debiased'] > 0.715) & (data['t04_spiral_a08_spiral_weight'] >= 20) & (data['t04_spiral_a08_spiral_debiased'] > 0.619)
    notedgeon = (data['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (data['t02_edgeon_a05_no_debiased'] > 0.715) & (data['t02_edgeon_a05_no_weight'] >= 20)

    sfr_sample = data[sf & redshift]

    return sfr_sample

def get_mass_sfr(sfr_sample):

    mass_sfr_good = (sfr_sample['MEDIAN_MASS'] > -99) & (sfr_sample['MEDIAN_SFR'] > -99)

    mass = sfr_sample[mass_sfr_good]['MEDIAN_MASS']
    sfr = sfr_sample[mass_sfr_good]['MEDIAN_SFR']

    return mass,sfr

def bins():

    mass_bins = np.linspace(6,13,50)
    sfr_bins = np.linspace(-5,2,50)

    return mass_bins,sfr_bins
    
def plot_ms_arms_number(sfr_sample):

    # Plot

    fig = plt.figure(1,(6,6))
    fig.clf()

    fig.subplots_adjust(left=0.08,bottom=0.15,right=0.9,hspace=0,wspace=0)

    mass,sfr = get_mass_sfr(sfr_sample)
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
        ax.set_ylim(-4,2)

        if idx == 4:
            ax.set_xlabel('Stellar mass [log '+r'$M/M_\odot$]',fontsize=16)
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

    box = ax.get_position()
    axColorbar = plt.axes([box.x0*1.05 + box.width * 0.95, box.y0, 0.01, box.height*2])
    cb = plt.colorbar(h[3],cax = axColorbar, orientation="vertical")
    cb.set_label(r'$N_\mathrm{star-forming\/galaxies}$' ,fontsize=16)

    fig.savefig('%s/ms_arms_number.pdf' % ms_path, dpi=200)

    return None

def plot_ms_bars(sfr_sample):

    # Plot

    fig = plt.figure(2)
    fig.clf()

    mass,sfr = get_mass_sfr(sfr_sample)
    mass_bins,sfr_bins = bins()
    h,xedges,yedges = np.histogram2d(mass,sfr,bins=(mass_bins,sfr_bins))

    notedgeon = (sfr_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sfr_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sfr_sample['t02_edgeon_a05_no_weight'] >= 20)

    barred = sfr_sample[notedgeon & (sfr_sample['t03_bar_a06_bar_debiased'] >= 0.4)] 
    unbarred = sfr_sample[notedgeon & (sfr_sample['t03_bar_a06_bar_debiased'] < 0.4)] 

    bar_data = (barred,unbarred)
    color = ('blue','red')

    # Plot barred and unbarred
    for idx, (b,c) in enumerate(zip(bar_data,color)):

        ax = fig.add_subplot(1,2,idx+1)
        h = ax.hist2d(mass,sfr,bins=50,cmap = cm.gray_r, norm=LogNorm())
        ax.set_xlim(6,13)
        ax.set_ylim(-5,2)
        ax.set_xlabel('Stellar mass (log'+r'$M/M_\odot$)',fontsize=20)
        ax.set_ylabel('Star formation rate'+r'$M_\odot/yr$',fontsize=20)
        fig.colorbar(h[3], ax=ax)

        ax.scatter(b['MEDIAN_MASS'],b['MEDIAN_SFR'], s=2, color=c, marker='o')

    fig.tight_layout()
    fig.savefig('%s/ms_bar.pdf' % ms_path, dpi=200)


    return None
    
def plot_ms_arms_winding(sfr_sample):

    # Plot

    fig = plt.figure(3,(10,4))
    fig.clf()

    fig.subplots_adjust(left=0.08,bottom=0.15,hspace=0,wspace=0)

    mass,sfr = get_mass_sfr(sfr_sample)
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
            ax.set_xlabel('Stellar mass [log '+r'$M/M_\odot$]',fontsize=16)
        #fig.colorbar(h[3], ax=ax)

        if idx > 0:
            ax.get_yaxis().set_ticks([])

        spiral = (sfr_sample['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & (sfr_sample['t02_edgeon_a05_no_debiased'] > 0.715) & (sfr_sample['t04_spiral_a08_spiral_weight'] >= 20) & (sfr_sample['t04_spiral_a08_spiral_debiased'] > 0.619)
        n1 = sfr_sample[spiral & (sfr_sample['t10_arms_winding_%s_flag' % a] == 1)] 
        ax.scatter(n1['MEDIAN_MASS'],n1['MEDIAN_SFR'], s = 2, color=c, marker='o')

        ax.text(6.2,1.3,r'$\phi_{arms} = $%s' % a[4:], color='k')

    # Set the colorbar and labels at the end

    box = ax.get_position()
    axColorbar = plt.axes([box.x0*1.05 + box.width * 0.95, box.y0, 0.01, box.height])
    cb = plt.colorbar(h[3],cax = axColorbar, orientation="vertical")
    cb.set_label(r'$N_\mathrm{star-forming\/galaxies}$' ,fontsize=16)

    #fig.tight_layout()
    fig.savefig('%s/ms_arms_winding.pdf' % ms_path, dpi=200)

    return None

