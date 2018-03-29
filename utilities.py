import numpy as np
import healpy as hp

import astropy.units as u
from astropy.coordinates import SkyCoord

import sys
if sys.platform=='darwin':
    top_path='/Users/codydirks/PGCC/'
else:
    top_path='/DataDisk/datafiles/PGCC_HST/'

x1d_dir=top_path+'x1d_files/'

# Returns the velocity shift necessary to convert Helio to LSR
# i.e. returns (V_lsr - V_helio)
def HelioToLSR(target_ra_deg, target_dec_deg):
    Vsun=19.7
    a0=271.0*np.pi/180.
    d0=30.0*np.pi/180.
    a=target_ra_deg*np.pi/180.
    d=target_dec_deg*np.pi/180.
    #shift=Vsun*(np.cos(a-a0)*np.cos(a0)*np.cos(a)+np.sin(d0)*np.sin(d))
    shift=Vsun*(np.cos(a0)*np.cos(d0)*np.cos(a)*np.cos(d)
                +np.sin(a0)*np.cos(d0)*np.sin(a)*np.cos(d)
                +np.sin(d0)*np.sin(d))
    return shift

# Calculates the angular distance between the center of a
# PGCC source and a given RA/Dec, returns results in units
# of the source's 2D elliptical Gaussian fit parameters.
def calc_r_dist(pgcc,sightline_ra,sightline_dec):
    angle=pgcc['gau_position_angle']
    sin=np.sin(-angle)
    cos=np.cos(-angle)
    a=pgcc['gau_major_axis']/2.
    b=pgcc['gau_minor_axis']/2.
    gal=SkyCoord(ra=sightline_ra*u.degree,dec=sightline_dec*u.degree,frame='icrs').galactic
    del_l=60.*(gal.l.value-pgcc['glon'])
    del_b=60.*(gal.b.value-pgcc['glat'])
    return np.sqrt((((cos*del_l+sin*del_b)/a)**2+((sin*del_l-cos*del_b)/b)**2))

# For a given sightline and atomic line, finds the spectral data in the file system
# and calculates the upper limit column density based on the S/N of that data
def get_ul(sl,line):
    f=get_atomic_entry(line)
    fil=x1d_dir+sl+'/E140H/'+sl+'_'+str(line)+'.dat'
    wavs=[]
    flxs=[]
    with open(fil,'r') as myfile:
        for entry in myfile:
            wav,flx=entry.split()[:2]
            wavs.append(float(wav))
            flxs.append(float(flx))

    wavs=np.array(wavs)
    flxs=np.array(flxs)
    flxcut=np.copy(flxs)
    n=0
    ind=np.array([])
    while abs(np.mean(flxcut)-1)>0.002:
        n=n+1
        ind = np.argpartition(abs(flxs-np.mean(flxcut)), -n)[-n:]
        flxcut=np.delete(flxs,ind)
    tau=np.log10(1/(1-5.*np.std(flxcut)))
    ul=3.768E14*tau/(f*line)
    return np.log10(ul)

# Parses the FITS6P atomic.dat file to find relevant line info
# for an atomic line of a given wavelength
def get_atomic_entry(wav):
    if wav==1328.833:
        wav=1328.8333
    with open('/Users/codydirks/fits6p/atomic.dat','r') as myfile:
        lines=myfile.read().split('\n')[1:]
    for line in lines:
        if float(line.split()[0])==wav:
            return float(line[30:39])


# Gets spectrum from the pixel that contains a given Lat/Lon
# from the Harvard CfA CO survey
def get_cfa_vels_spec(data,hdr,glon,glat):
    cfa_vels=hdr['CRVAL1']+hdr['CDELT1']*np.arange(hdr['NAXIS1'])
    cfa_glons=hdr['CRVAL2']+hdr['CDELT2']*np.arange(hdr['NAXIS2'])
    cfa_glats=hdr['CRVAL3']+hdr['CDELT3']*np.arange(-hdr['NAXIS3']/2+1,hdr['NAXIS3']/2+1)

    if glon > 180:
        glon=glon-360
    glon_idx=min(range(len(cfa_glons)),key=lambda i: abs(cfa_glons[i]-glon))
    glat_idx=min(range(len(cfa_glats)),key=lambda i: abs(cfa_glats[i]-glat))
    cfa_spec=data[glat_idx,glon_idx,:]
    return cfa_vels,cfa_spec

# Gets a CO map based on a given GLON/GLAT and box size.
# Uses vel and vel_width parameters to choose velocity range,
# then integrates along the velocity axis to determine the 2D map
def get_co_map(data,hdr,glon,glat,box,vel=0,vel_width=20):
    cfa_vels=hdr['CRVAL1']+hdr['CDELT1']*np.arange(hdr['NAXIS1'])
    cfa_glons=hdr['CRVAL2']+hdr['CDELT2']*np.arange(hdr['NAXIS2'])
    cfa_glats=hdr['CRVAL3']+hdr['CDELT3']*np.arange(-hdr['NAXIS3']/2+1,hdr['NAXIS3']/2+1)

    if glon > 180:
        glon=glon-360

    min_vel=vel-vel_width
    max_vel=vel+vel_width
    min_vel_idx=min(range(len(cfa_vels)), key=lambda i: abs(cfa_vels[i]-min_vel))
    max_vel_idx=min(range(len(cfa_vels)), key=lambda i: abs(cfa_vels[i]-max_vel))

    # Isolate a box around the sightline
    box=2.0
    glon_idx=min(range(len(cfa_glons)),key=lambda i: abs(cfa_glons[i]-glon))
    glat_idx=min(range(len(cfa_glats)),key=lambda i: abs(cfa_glats[i]-glat))
    min_glon=min(range(len(cfa_glons)),key=lambda i: abs(cfa_glons[i]-(glon+box/2.)))
    max_glon=min(range(len(cfa_glons)),key=lambda i: abs(cfa_glons[i]-(glon-box/2.)))
    min_glat=min(range(len(cfa_glats)),key=lambda i: abs(cfa_glats[i]-(glat-box/2.)))
    max_glat=min(range(len(cfa_glats)),key=lambda i: abs(cfa_glats[i]-(glat+box/2.)))
    #print cfa_data[min_vel_idx:max_vel_idx,min_glon:max_glon,min_glat:max_glat]
    cube=data[min_glat:max_glat,min_glon:max_glon,min_vel_idx:max_vel_idx]
    cube[np.isnan(cube)]=0
    dat=np.sum(cube,axis=2)
    #if cfa_glons[min_glon] < 0:
    #    return dat[::-1,:],[cfa_glons[min_glon]+360,cfa_glons[max_glon]+360,cfa_glats[min_glat],cfa_glats[max_glat]]
    #else:
    #return dat[::-1,:],[cfa_glons[min_glon],cfa_glons[max_glon],cfa_glats[min_glat],cfa_glats[max_glat]]
    return dat[::-1,:],[1.1,-1.1,-1.1,1.1]
