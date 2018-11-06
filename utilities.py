import numpy as np
import os
from itertools import chain

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
def get_ul(sl,ion,line):
    f=get_atomic_entry(ion,line)
    fls=[x for x in os.listdir(x1d_dir+sl+'/E140H') if x.startswith(sl) and x.endswith('.dat')]
    fls.sort(key=lambda x:abs(line-float(x.split('_')[1][:-4])))
    fil=x1d_dir+sl+'/E140H/'+fls[0]
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

    # Try to remove absorption lines from spectrum
    # so our calculation of tau only involves the
    # noise envelope
    while abs(np.mean(flxcut)-1)>0.002:
        n=n+1
        ind = np.argpartition(abs(flxs-np.mean(flxcut)), -n)[-n:]
        flxcut=np.delete(flxs,ind)
        if len(flxcut)==0:
            # If we weren't able to converge, reset the selected data
            # to the original spectrum
            flxcut=np.copy(flxs)
            break

    tau=np.log10(1/(1-5.*np.std(flxcut)))
    ul=3.768E14*tau/(f*line)
    return np.log10(ul)



def get_f_star(element_data, element_params, sightline, list_of_elements):
    el1=list_of_elements[0]
    el1row=element_data[(element_data['HD']==sightline) & (element_data['El']==el1)]
    el1param=element_params[element_params['El']==el1].iloc[0]
    el1n=el1row['logNx']
    el1_frac_err=(el1row['B_logNx']-el1row['b_logNx'])/2.
    el1err=np.log10(el1_frac_err/0.434)+el1n
    f_vals=np.empty(0)
    f_errs=np.empty(0)
    for el2 in list_of_elements[1:]:
        el2row=element_data[(element_data['HD']==sightline) & (element_data['El']==el2)]
        el2param=element_params[element_params['El']==el2].iloc[0]
        if len(el2row)>0:
            el2row=el2row.iloc[0]
            el2n=el2row['logNx']
            el2_frac_err=(el2row['B_logNx']-el2row['b_logNx'])/2.
            el2err=np.log10(el2_frac_err/0.434)+el2n
            f,f_err=calc_f(el1param,el2param,el1n,el2n,el1err,el2err)
            f_vals=np.concatenate([f_vals,[f]])
            f_errs=np.concatenate([f_errs,[f_err]])
    return np.mean(f_vals),np.sqrt(np.sum(f_errs**2))

# Calculates the F* parameter defined by Jenkins(2009) given data
# for two particular elements
# Inputs:   el1row, el2row are parameter rows from Jenkins09 Table4
#           el1n,el2n,el1err,el2err are log(N) and associated errors for the two elements
def calc_f(el1row,el2row,el1n,el2n,el1err,el2err):
    el1n=float(el1n)
    el2n=float(el2n)
    el1err=float(el1err)
    el2err=float(el2err)
    el1_frac_err=0.434*10**(el1err-el1n)
    el2_frac_err=0.434*10**(el2err-el2n)
    fs=np.arange(-1,3,0.01)
    a1,a1err,b1,b1err,x1,x1err=chain.from_iterable([[el1row[idx],el1row['e_'+idx]] for idx in ('Ax','Bx','[X/H]')])
    z1=el1row['zx']
    a2,a2err,b2,b2err,x2,x2err=chain.from_iterable([[el2row[idx],el2row['e_'+idx]] for idx in ('Ax','Bx','[X/H]')])
    z2=el1row['zx']
    el1_depls=b1+a1*(fs-z1)
    el2_depls=b2+a2*(fs-z2)

    rats=el1_depls-el2_depls
    rat=el1n-el2n-(x1-12)+(x2-12)
    f=fs[np.abs(rats-rat).argmin()]

    f_err=np.sqrt((el1_frac_err**2+el2_frac_err**2
                    #+x1err**2+x2err**2
                    #+b1err**2+b2err**2
                    #+(z1*(a1-a2)*a1err)**2 + (z2*(a1-a2)*a2err)**2
                    #+((a1*z1)**2+(a2*z2)**2)*(a1err**2+a2err**2)
                  )/(a1-a2)**2
                 )
    return f,f_err

# Parses the FITS6P atomic.dat file to find relevant line info
# for an atomic line of a given wavelength
def get_atomic_entry(ion,wav):
    if wav==1328.833:
        wav=1328.8333
    with open('/Users/codydirks/fits6p/atomic.dat','r') as myfile:
        lines=myfile.read().split('\n')[1:]
    if 'CO' in ion:
        ion_lines=[x for x in lines if x[10:18].strip().startswith(ion)]
    else:
        ion_lines=[x for x in lines if x[10:18].strip()==ion]
    ion_lines.sort(key=lambda x: abs(wav-float(x[0:8])))
    return float(ion_lines[0][30:39])


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
