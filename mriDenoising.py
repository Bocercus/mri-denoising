"""
Denoising functions intended for MRI
Author: Niklas S. Truelsen
"""
import numpy as np
import scipy.linalg as scla

def deMosaic(dat,depthx,depthy,my_b):
    datm=[]
    picx=int(dat.shape[0]/depthx)
    picy=int(dat.shape[1]/depthy)
    for x in range(depthx):
        for y in range(depthy):
            datm.append(dat[picx*x:picx*(x+1),picy*y:picy*(y+1),my_b])
    return np.array(datm)

def mse(xarr,yarr):
    """Mean square error
    (numpy array, numpy array)"""
    return sum((xarr-yarr)**2)/len(xarr)

def submatwrap(mat,xx,yy,tt): #Find submatrx with center (xx,yy) with periodic boundary conditions
    return np.roll(mat,(tt-xx,tt-yy),(0,1))[:2*tt+1,:2*tt+1]

def submatwrapdim(mat,c,tt): #Find submatrix with corner (xx,yy) with periodic boundary conditions
    for i in range(len(c)):
        if(len(mat.shape)>1):
            mat=np.swapaxes(mat,0,i)
        mat=np.roll(mat,tt-c[i],0)[:2*tt+1]
    if(len(mat.shape)>1):
        for i in range(len(c)):
            mat=np.swapaxes(mat,i,1)
    return mat

def submatcorn(mat,xx,yy,tt): #Submatrix with corner origin
    return mat[(xx):(xx+tt),(yy):(yy+tt)]

def submatdimcorn(mat,c,tt): #c-dim submatrix with corner origin
    for i in range(len(c)):
        if(len(mat.shape)>1):
            mat=np.swapaxes(mat,0,i)
        mat=mat[(c[i]):(c[i]+tt)]
    if(len(mat.shape)>1):
        for i in range(len(c)):
            mat=np.swapaxes(mat,i,1)
    return mat

def medianfilter(mat,r):
    """
    Median prefilter 2D
    (numpy matrix, int radius)
    """
    mx=mat.shape[0]
    my=mat.shape[1]
    outp=np.zeros(mat.shape)
    for xs in range(mx):
        for ys in range(my):
            outp[xs][ys]=np.median(submatwrap(mat,xs,ys,r).reshape(-1))
    return outp

def medianfilter3D(mat,r):
    """
    Median prefilter 3D
    (numpy matrix, int radius)
    """
    mx=mat.shape[0]
    my=mat.shape[1]
    mz=mat.shape[2]
    outp=np.zeros(mat.shape)
    for xs in range(mx):
        for ys in range(my):
            for zs in range(mz):
                outp[xs][ys][zs]=np.median(submatwrapdim(mat,[xs,ys,zs],r).reshape(-1))
    return outp

def denoiseMatrix(X):
    """
    MP-PCA Denoising of numpy matrix X
    Ported from https://github.com/sunenj/MP-PCA-Denoising/blob/a89e20f381f92b0785d3b34ba914a4f0d2955f29/denoiseMatrix.m
    https://doi.org/10.1002/mrm.27658
    """
    assert type(X).__module__ == np.__name__, "Not numpy array"
    if len(X.shape)==1:
        #X=X.reshape(X,(1,len(X)))
        print("X 1-dim.")
        return X,-1,-1,-1,-1,True
    M=X.shape[0]
    N=X.shape[1]
    if M<N:
        lam,U=scla.eig( np.matmul(X, np.transpose(X)) )#py.eig: val,vec, matlab.eig: vec,val
        lam/=N
    else:
        lam,U=scla.eig( np.matmul(np.transpose(X), X) )
        lam/=M
    order=np.argsort(lam)[::-1]
    lam=np.sort(lam)[::-1]
    U=U[:,order]
    csum=np.cumsum(lam[::-1])[::-1]
    p=np.array(range(len(lam)))
    p=np.argwhere( np.multiply((lam-lam[-1]),np.multiply(M-p,N-p))< 4*csum*np.sqrt(M*N) ) #-1?
    #Niklas START (sometimes p is empty. Why?)
    if p.size==0:
        p=np.zeros(1)
    elif len(p.shape)>1:
        p=p[0]
    #if(type(p)!=type(2)):
        #p=p[0]
    #Niklas END
    s2div=np.multiply((M-p),(N-p))
    p=int(p[0])#p.tolist()
    if p==[0] or p==0:
        X=np.zeros(X.shape)
    elif M<N:
        X=np.matmul(np.matmul(U[:,:p],np.transpose(U[:,:p])),X)
    else:
        X=np.matmul(np.matmul(X,U[:,:p]),np.transpose(U[:,:p]))
    s2=csum[p]/s2div
    s2_after=s2-csum[p]/(M*N)
    return X,s2,p,s2_after,lam,False

class psr: #Patch similarity object
    def __init__(self,mser,Ac,Bc=np.zeros(2)):
        self.mser=mser #mean square error result
        self.Ac=Ac
        self.Bc=Bc
    def p(self): #Print function for debugging
        print(self.mser)
        print(self.Ac)
        print(self.Bc)

def paCsim2D(patch,tt,k=-1):
    """Find 2D patches similar to center patch, tt: width, k: amnt of res"""
    ci=int((patch.shape[0]-1)/2) #center index, w=2t+1 => (w-1)/2=t
    big_t=2*tt+1
    cenpan=submatcorn(patch,ci-tt,ci-tt,big_t) #center patch
    sedi=patch.shape[0] #search dimensions
    resarr=[] #result array
    for x in range(tt,sedi-2*tt+1):
        for y in range(tt,sedi-2*tt+1):
            if(x==ci and y==ci):
                continue #Comparison with center pixel and itself is unnecessary
            sema=submatcorn(patch,x-tt,y-tt,big_t) #search matrix
            resobj=psr(mse(cenpan.reshape(-1),sema.reshape(-1)),[x,y]) #save mse's in psr-object
            resarr.append(resobj)
    resarr.sort(key=lambda ob: ob.mser, reverse=False) #sort results by mser
    if k==-1:
        return resarr
    else:
        return resarr[:k]

def paCsim3D(patch,tt,k=-1):
    """Find 3D patches similar to center patch, tt: width, k: amnt of res"""
    ci=int((patch.shape[0]-1)/2) #center index, w=2t+1 => (w-1)/2=t
    big_t=2*tt+1
    cenpan=submatdimcorn(patch,[ci-tt,ci-tt,ci-tt],big_t) #center patch
    sedi=patch.shape[0] #search dimensions
    resarr=[] #result array
    for x in range(tt,sedi-2*tt+1):
        for y in range(tt,sedi-2*tt+1):
            for z in range(tt,sedi-2*tt+1):
                if(x==ci and y==ci and z==ci):
                    continue #Comparison with center pixel and itself is unnecessary
                sema=submatdimcorn(patch,[x-tt,y-tt,z-tt],big_t) #search matrix
                resobj=psr(mse(cenpan.reshape(-1),sema.reshape(-1)),[x,y,z]) #save mse's in psr-object
                resarr.append(resobj)
    resarr.sort(key=lambda ob: ob.mser, reverse=False) #sort results by mser
    if k==-1:
        return resarr
    else:
        return resarr[:k]

def fullDenoise(mat,k,t,r,medianpre=False,mrad=2):
    """
    Full 2d denoising function (Based on Manjón)
    numpy 2d array mat: picture, k: # of patches to keep, t: patch matrix size, r: window radius
    """
    mx=mat.shape[0]
    my=mat.shape[1]
    outp=np.zeros(mat.shape)#[::-1])
    dnfc=0 #denoise fail counter
    medpref=[] #prefiltered picture
    sml_t=int((t-1)/2) #small t
    big_t=t #big t, 2*t+1 #T=2t+1 => t=(T-1)/2
    if medianpre:
        medpref=medianfilter(mat,mrad)#Median filter, Coupe p. 851: 3x3x3 => mrad=1
    for xs in range(mx):
        for ys in range(my): #For each pixel
            if medianpre:
                grp=submatwrap(medpref,xs,ys,r) #find window submatrix
            else:
                grp=submatwrap(mat,xs,ys,r) #find window submatrix
            psres=paCsim2D(grp,sml_t,k) #find similar patches
            rec=submatcorn(grp,r-sml_t,r-sml_t,big_t).reshape(-1) #add center patch
            for myp in psres:
                newrow=submatcorn(grp,myp.Ac[0]-sml_t,myp.Ac[1]-sml_t,big_t).reshape(-1)
                if(len(newrow)==big_t*big_t):
                    rec=np.vstack([rec,newrow]) #add similar patches
            rec=np.transpose(np.array(rec))
            dM,dnA,dnB,dnC,_,dnerr=denoiseMatrix(rec)
            if dnerr:
                dnfc+=1
            if len(dM)>1:
                #print(rec[2*(t*t+t),0],dM[2*(t*t+t),0])
                outp[xs,ys]=dM[2*(sml_t*sml_t+sml_t),0] #(2t+1)*t+t=2(t^2+t) <= center index
    return outp,dnfc

def fullDenoise3D(mat,k,t,r,medianpre=False,mrad=1):
    """
    Full 2d denoising function (Based on Manjón)
    numpy 2d array mat: picture, k: # of patches to keep, t: patch matrix size, r: window radius
    """
    mx=mat.shape[0]
    my=mat.shape[1]
    mz=mat.shape[2]
    outp=np.zeros(mat.shape)#[::-1])
    dnfc=0 #denoise fail counter
    medpref=[] #prefiltered picture
    sml_t=int((t-1)/2) #small t
    big_t=t #big t, 2*t+1 #T=2t+1 => t=(T-1)/2
    if medianpre:
        medpref=medianfilter3D(mat,mrad)#Median filter, Coupe p. 851: 3x3x3 => mrad=1
    for xs in range(mx):
        for ys in range(my): #For each pixel
            for zs in range(mz):
                if medianpre:
                    grp=submatwrapdim(medpref,[xs,ys,zs],r) #find window submatrix
                else:
                    grp=submatwrapdim(mat,[xs,ys,zs],r) #find window submatrix
                psres=paCsim3D(grp,sml_t,k) #find similar patches
                rec=submatdimcorn(grp,[r-sml_t,r-sml_t,r-sml_t],big_t).reshape(-1) #add center patch
                for myp in psres:
                    newrow=submatdimcorn(grp,[myp.Ac[0]-sml_t,myp.Ac[1]-sml_t,myp.Ac[2]-sml_t],big_t).reshape(-1)
                    if(len(newrow)==big_t*big_t*big_t):
                        rec=np.vstack([rec,newrow]) #add similar patches
                rec=np.transpose(np.array(rec))
                dM,dnA,dnB,dnC,_,dnerr=denoiseMatrix(rec)
                if dnerr:
                    dnfc+=1
                elif len(dM)>1:
                    #print(rec[2*(t*t+t),0],dM[2*(t*t+t),0])
                    #print(rec.shape,dM.shape,rec)
                    outp[xs,ys,zs]=dM[4*(sml_t*sml_t*sml_t)+6*(sml_t*sml_t)+3*sml_t,0]
                    #For 2D: T*t+t=(2t+1)*t+t=2(t^2+t) <= center index
                    #For 3D: T*T*t+T*t+t=(2t+1)^2*t+(2t+1)*t+t=4t^3+6t^2+3t
    return outp,dnfc