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
    gaia_filenames=[top_path+'gaia_data/TgasSource_000-000-0'+'{:02}'.format(i)+'.fits' for i in range(16)]
    pgcc_data=fits.open(top_path+'HFI_PCCS_GCC_R2.02.fits')[1].data
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
                gaia_entry=fits.open(gaia_filenames[fl])[1].data[idx]
            else:
                gaia_entry=None

            sl_pgcc_gaia.append([sightline,(ra,dec),pgcc,gaia_entry])
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
                    comps_dict[ion]=[c for c in entry[1] if abs(c.v-v)<del_v_tol]
                    if len(comps_dict[ion])>0:
                        n_tot=sum([x.n for x in comps_dict[ion]])
                        n_err=np.sqrt(sum([x.n_err*x.n_err for x in comps_dict[ion]]))
                        n_dict[ion]=round(np.log10(n_tot),3)
                        #n_dict[ion+'_err']=0.434*n_err/n_tot
                        n_dict[ion+'_err']=round(np.log10(n_err),3)
                        break
                    else:
                        fits6p_prefix=get_fits6p_ion(ion)
                        for fl in [x for x in os.listdir(x1d_dir+sl[0]+'/E140H') if x.startswith(fits6p_prefix) and x.endswith('.txt')]:
                            lam=fl.split('_')[1]
                            dat_file=[x for x in os.listdir(x1d_dir+sl[0]+'/E140H') if x.startswith(sl[0]) and x.endswith('.dat') and lam in x][0]
                            lin=dat_file.split('_')[1][:-4]
                            n_dict[ion+'_err']=get_ul(sl[0],float(lin))
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
    all_data['H_tot']=np.where(all_data['O']>0,np.log10((10**6/305.)*10**(all_data['O'])),0).round(3)
    all_data['H_2']=np.where(all_data['Cl']>0, ((all_data['Cl']+3.7)/0.87),0.).round(3)
    all_data['f_H2']=np.where((all_data['H_tot']>0) & (all_data['H_2']>0),10**(np.log10(2.)+all_data['H_2']-all_data['H_tot']),0.0).round(3)
    all_data['CO/H2']=np.where((all_data['H_2']>0) & (all_data['CO']>0),10**(all_data['CO']-all_data['H_2']),0.0)
    all_data['12CO/13CO']=np.where((all_data['13CO']>0) & (all_data['CO']>0),10**(all_data['CO']-all_data['13CO']),0.0).round(3)

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
