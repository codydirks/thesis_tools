class LineInfo(object):
    # Stores information from FITS6P .txt output files as objects
    def __init__(self,dat_line,err_line):
        # FITS6P outputs two lines for each velocity component: the
        # data line containing line parameters and flags, and an error
        # line containing 1sigma values for each physical parameter
        dat=dat_line.split()
        if len(dat)==10:
            self.ion=dat[0]
            self.n=float(dat[1])
            self.b=float(dat[2])
            self.wav=float(dat[3])
            self.v=float(dat[4])
            self.n_flag=int(dat[5])
            self.b_flag=int(dat[6])
            self.v_flag=int(dat[7])
            self.eqw=float(dat[8])
            self.tot_eqw=float(dat[9])
        elif len(dat)<10:
            self.ion=dat_line[0:10]
            self.n=float(dat_line[10:20])
            self.b=float(dat_line[20:28])
            self.wav=float(dat_line[28:38])
            self.v=float(dat_line[38:48])
            self.n_flag=int(dat_line[48:52])
            self.b_flag=int(dat_line[52:54])
            self.v_flag=int(dat_line[54:56])
            self.eqw=float(dat_line[56:64])
            self.tot_eqw=float(dat_line[64:])

        errs=err_line.split()
        self.n_err=float(errs[0])
        self.b_err=float(errs[1])
        self.v_err=float(errs[2])

    def __repr__(self):
        return self.ion
    def __str__(self):
        return self.ion
