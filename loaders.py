import re
import os
import pandas as pd
import numpy as np

import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord

from thesis_tools.LineInfo import LineInfo
from thesis_tools.utilities import calc_r_dist,get_ul
from itertools import chain

import sys
if sys.platform=='darwin':
    top_path='/Users/codydirks/PGCC/'
else:
    top_path='/DataDisk/datafiles/PGCC_HST/'

x1d_dir=top_path+'x1d_files/'

c=300000.


# Collates various info about each sightline-PGCC pair
# Returns a data structure containing all sightline info, or a given selection
# Each entry in the returned data has the form:
# Sightline name, coordinate tuple, pgcc data, gaia data
def load_data(selection=[]):
    filename='sightline_pgcc_gaia_results.txt'
    tgas_filenames=[top_path+'tgas_data/TgasSource_000-000-0'+'{:02}'.format(i)+'.fits' for i in range(16)]
    pgcc_hdu=fits.open(top_path+'HFI_PCCS_GCC_R2.02.fits')
    pgcc_data=pgcc_hdu[1].data
    pgcc_hdu.close()
    sl_pgcc_gaia=[]
    with open(filename,'r') as myfile:
        for line in myfile:
            dat=line.strip('|\n').split('|')
            sightline=dat[0]
            coords=dat[1].split(' ')
            ra=float(coords[0])*u.degree
            dec=float(coords[1])*u.degree
            pgcc=pgcc_data[int(dat[2])]
            if dat[3] != 'None':
                fl,idx=map(int,dat[3][1:-1].split(','))
                tgas_hdu=fits.open(tgas_filenames[fl])
                tgas_entry=tgas_hdu[1].data[idx]
                tgas_hdu.close()
            else:
                tgas_entry=None


            sl_pgcc_gaia.append([sightline,(ra,dec),pgcc,tgas_entry])
    sl_pgcc_gaia.sort(key=lambda x: (x[0][0],int(re.search(r'\d+',x[0]).group())))

    if type(selection)==list and len(selection)>0:
        return [x for x in sl_pgcc_gaia if x[0] in selection]
    elif type(selection) is str:
        return [x for x in sl_pgcc_gaia if x[0]==selection][0]
    else:
        return sl_pgcc_gaia

# Loads all of my FITS6P results into a Pandas dataframe for a set
# of given sightlines.
# Input: List of entries from the load_data function
def load_results(primary_sample):
    del_v_tol=1.5
    ion_list=['O_I]','Cl_I','C_I','C_I*','C_I**','CO','13CO','Ni_II','Kr_I','Ge_II','Mg_II']
    ns_dict={}
    info={'Sightline':[],'R_dist':[],'Velocity':[]}
    for ion in ion_list:
        ns_dict[ion]=0
        info[get_table_ion(ion)]=[]
        info[get_table_ion(ion)+'_err']=[]

    all_data=pd.DataFrame(info)

    for sl in primary_sample:
        info['Sightline']=sl[0]
        ra,dec=sl[1]
        galcoords=SkyCoord(ra=ra,dec=dec).galactic
        pgcc=sl[2]
        info['R_dist']=round(calc_r_dist(pgcc,ra.value,dec.value),2)

        data_dict={}
        for ion in ion_list:
            data_dict[ion]=get_sightline_fits6p_results(info['Sightline'],ion)

        vel_comps=[]
        [vel_comps.append(v) for v in [y.v for y in chain(*[x[1] for x in (data_dict['CO']+data_dict['13CO']+data_dict['O_I]']+data_dict['Cl_I'])])]
         if any([abs(v-z)<del_v_tol for z in vel_comps])==False]

        vel_comps.sort()
        for v in vel_comps:
            n_dict={}
            comps_dict={}
            # Initialize components and columns to zero
            for ion in ion_list:
                n_dict[ion]=0
                n_dict[ion+'_err']=0
                comps_dict={}

            # Iterate over ions and get nearest component, then add columns in that component
            for ion in ion_list:
                for entry in data_dict[ion]:
                    comps_dict[ion]=[x for x in entry[1] if abs(x.v-v)<del_v_tol]
                    if len(comps_dict[ion])>0:
                        n_tot=sum([x.n for x in comps_dict[ion]])
                        n_err=np.sqrt(sum([x.n_err*x.n_err for x in comps_dict[ion]]))
                        n_dict[ion]=round(np.log10(n_tot),3)
                        #n_dict[ion+'_err']=0.434*n_err/n_tot
                        if n_err>0:
                            n_dict[ion+'_err']=round(np.log10(n_err),3)
                        else:
                            lin=comps_dict[ion][0].wav/(comps_dict[ion][0].v/c + 1)
                            n_dict[ion+'_err']=get_ul(sl[0],ion,float(lin))
                        break
                    else:
                        fits6p_prefix=get_fits6p_ion(ion)
                        for fl in [x for x in os.listdir(x1d_dir+sl[0]+'/E140H') if x.startswith(fits6p_prefix) and x.endswith('.txt')]:
                            lam=fl.split('_')[1]
                            dat_file=[x for x in os.listdir(x1d_dir+sl[0]+'/E140H') if x.startswith(sl[0]) and x.endswith('.dat') and lam in x][0]
                            lin=dat_file.split('_')[1][:-4]
                            n_dict[ion+'_err']=get_ul(sl[0],ion,float(lin))
                        break

                info[get_table_ion(ion)]=n_dict[ion]
                info[get_table_ion(ion)+'_err']=n_dict[ion+'_err']
            info['Velocity']=v
            new_row=pd.DataFrame({key:[info[key]] for key in info})
            all_data=pd.concat([all_data,new_row])

    cols=['Sightline','R_dist','Velocity']+list(chain.from_iterable([[get_table_ion(i),get_table_ion(i)+'_err'] for i in ion_list]))
    all_data=all_data[cols].reset_index(drop=True)
    all_data=all_data.replace(-np.inf,0)

    # All calculated columns

    # Derive the total hydrogen column density based on the oxygen column density
    # using the relationship from (Meyer et al. 1998) GHRS data

    #Converts log(oxygen column density) to log(total hydrogen column density)
    def o_to_h_tot(row):
        log_o,log_o_err=row[['O','O_err']]
        o,o_err=10**np.array((log_o,log_o_err))
        log_h_tot,log_h_tot_err=0.,0.
        if log_o>0:
            log_h_tot=np.log10((10**6/319.)*o).round(3)
            log_h_tot_err=np.log10((10**log_h_tot)*np.sqrt((14./319.)**2+(o_err/o)**2)).round(3)
        else:
            log_h_tot_err=np.log10((10**6/319.)*o_err).round(3)
        return log_h_tot,log_h_tot_err

    all_data['H_tot'],all_data['H_tot_err']=zip(*all_data.apply(o_to_h_tot,axis=1))


    # Determines the H2 column density. If CO is detected, use CO/CI ratio
    # Else, use chlorine column density a la Balashev
    def calc_h2(row):
        if row['CO']>0:
            return co_cI_to_h2(row)
        else:
            return cl_to_h2(row)

    # Derives the molecular hydrogen column density using the Balashev et al. 2015 relationship
    def cl_to_h2(row):
        log_cl,log_cl_err=row[['Cl','Cl_err']]
        cl,cl_err=10**np.array((log_cl,log_cl_err))
        log_h2,log_h2_err=0.,0.
        if log_cl>0:
            log_h2=(log_cl+3.7)/0.87
            # Error in Balashev paper is given as "about 0.2 dex"
            log_h2_err=np.log10((0.2/0.434)*10**log_h2)
        return round(log_h2,3),round(log_h2_err,3)

    # Calculates H2 column density based on CO/CI ratio.
    # This is only possible if a CO column density is measured
    def co_cI_to_h2(row):
        # These fit parameters are derived based on the combination of Burgh 2010 + my dataset
        m=0.949756
        dm=0.05089
        b=5.4674
        db=0.3079

        log_co,log_co_err=row[['CO','CO_err']]
        log_h2,log_h2_err=0.,0.
        if log_co>0:
            co_frac_err=10**(log_co_err-log_co)
            log_c=np.log10((10**row[['C','C*','C**']]).sum())
            log_c_err=np.log10(np.sqrt(((10**row[['C_err','C*_err','C**_err']])**2).sum()))
            c_frac_err=10**(log_c_err-log_c)
            log_co_c_ratio=log_co-log_c
            log_co_h2_ratio=((log_co_c_ratio)-b)/m

            log_h2=round(log_co-log_co_h2_ratio,3)
            d_co_h2_ratio=(1/m**2)*((0.434*(co_frac_err**2+c_frac_err**2))**2+db**2)+(((b-log_co_c_ratio)/m**2)**2)*dm**2
            h2_frac_err=np.sqrt(abs((d_co_h2_ratio/0.434)**2-co_frac_err**2))
            log_h2_err=np.log10((10**log_h2)*h2_frac_err).round(3)
        return log_h2,log_h2_err

    all_data['H2'],all_data['H2_err']=zip(*all_data.apply(calc_h2,axis=1))

    # Calculates the molecular hydrogen fraction using the above results
    def calc_fh2(row):
        log_h2,log_h2_err=row[['H2','H2_err']]
        h2,h2_err=10**np.array((log_h2,log_h2_err))
        log_h_tot,log_h_tot_err=row[['H_tot','H_tot_err']]
        h_tot,h_tot_err=10**np.array((log_h_tot,log_h_tot_err))
        f_h2,f_h2_err=0.,0.
        if log_h2>0 and log_h_tot>0:
            f_h2=(2*h2)/h_tot
            f_h2_err=f_h2*np.sqrt((h_tot_err/h_tot)**2+(h2_err/h2)**2)
        return f_h2,f_h2_err

    all_data['f_H2'],all_data['f_H2_err']=zip(*all_data.apply(calc_fh2,axis=1))

    return all_data

# Searches through file system to find FITS6P .txt output files for
# the specific sightline and ion listed
def get_sightline_fits6p_results(sightline, ion):
    del_v_tol=1.5
    data=[]
    fits6p_prefix=get_fits6p_ion(ion)
    for fl in [x for x in os.listdir(x1d_dir+sightline+'/E140H') if x.startswith(fits6p_prefix) and x.endswith('.txt')]:
        ions=parse_results_file(x1d_dir+sightline+'/E140H/'+fl)
        if 'CO' in ion:
            selected_ions=[i for i in ions if i.ion.startswith(ion)]
        else:
            selected_ions=[i for i in ions if i.ion==ion]
        data.append((fl,selected_ions))
    return data

# Stores results from a FITS6P .txt file in a list of LineInfo objects
def parse_results_file(filename):
    ids=['O_I]','Cl_I','C_I','C_I*','C_I**','CO','13CO','Ni_II','Kr_I','Ge_II','Mg_II']
    ions=[]
    with open(filename) as myfile:
        lines=[x.strip() for x in myfile.readlines()]
        for i in range(len(lines)):
            if len(lines[i].strip())>0:
                dat=lines[i].split()
                if any([dat[0]==x for x in ids]) or dat[0].startswith('CO_'):
                    ions.append(LineInfo(lines[i],lines[i+1]))
    return ions


# Translates ion strings to coincide with FITS6P table results
def get_fits6p_ion(strng):
    return strng.strip('*').replace('_','').lower().replace('i','I').replace('nI','ni').strip(']')

# Translate ion strings to conincide with table headers
def get_table_ion(strng):
    return strng.replace('_I]','').replace('_II','').replace('_I','')
