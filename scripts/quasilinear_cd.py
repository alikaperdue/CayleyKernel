#!/usr/bin/env python3
"""Independent NumPy implementation of Lemley Algorithm 3.5 (2609.11588v1).

Real inputs, arbitrary matching batch axes, CD coefficients on the last axis.
This is an experimental CPU companion, not a replacement for CayleyKernel.
The central complex coefficient is NOT a CD imaginary basis unit. Complex
pairings are bilinear (no coefficient conjugation). Recursive calls share
one complex coefficient field at all levels.
"""
import numpy as np


def complex_product(a,b):
    """Three real input-dependent products, equation (3.5)."""
    p=a.real*b.real;q=a.imag*b.imag
    r=(a.real+a.imag)*(b.real+b.imag)
    return (p-q)+1j*(r-p-q)


def complex_cross(X,Y,stats=None):
    N=X.shape[-1]
    if N==2:return np.zeros_like(X,dtype=np.complex128)
    h=N//2;m=h-1
    if stats is not None:
        stats['complex_nodes']+=1
        stats['real_bilinear_products_per_pair']+=18*m
    u,v=X[...,:h],Y[...,:h]
    gamma,delta=X[...,h:h+1],Y[...,h:h+1]
    s=X[...,h:].copy();s[...,0]=0
    t=Y[...,h:].copy();t[...,0]=0
    if N==4:
        Zp=Zm=None
    else:
        Zp=complex_cross(u+1j*s,v+1j*t,stats)
        Zm=complex_cross(u-1j*s,v-1j*t,stats)
    F=np.zeros_like(u,dtype=np.complex128);G=np.zeros_like(u,dtype=np.complex128)
    F[...,1:]=complex_product(gamma,t[...,1:])-complex_product(delta,s[...,1:])
    G[...,1:]=complex_product(delta,u[...,1:])-complex_product(gamma,v[...,1:])
    G[...,0]=np.sum(complex_product(s[...,1:],v[...,1:]),axis=-1)-np.sum(complex_product(t[...,1:],u[...,1:]),axis=-1)
    if Zp is not None:
        F[...,1:]+=(Zp[...,1:]+Zm[...,1:])/2
        G[...,1:]+=1j*(Zp[...,1:]-Zm[...,1:])/2
    return np.concatenate((F,G),axis=-1)


def real_cross(X,Y,stats=None):
    N=X.shape[-1]
    if N==2:return np.zeros_like(X,dtype=np.float64)
    h=N//2;m=h-1
    if stats is not None:
        stats['real_root_nodes']+=1
        stats['real_bilinear_products_per_pair']+=6*m
    u,v=X[...,:h],Y[...,:h]
    gamma,delta=X[...,h:h+1],Y[...,h:h+1]
    s=X[...,h:].copy();s[...,0]=0
    t=Y[...,h:].copy();t[...,0]=0
    Z=complex_cross(u+1j*s,v+1j*t,stats)
    F=Z.real+gamma*t-delta*s
    G=-Z.imag+delta*u-gamma*v
    G[...,0]=np.sum(s[...,1:]*v[...,1:],axis=-1)-np.sum(t[...,1:]*u[...,1:],axis=-1)
    return np.concatenate((F,G),axis=-1)


def multiply(a,b,stats=None):
    a,b=np.asarray(a),np.asarray(b)
    if a.ndim<1 or b.ndim<1 or a.shape[-1]!=b.shape[-1]:raise ValueError('matching coefficient axes required')
    N=a.shape[-1]
    if N<1 or N&(N-1):raise ValueError('dimension must be a positive power of two')
    if np.iscomplexobj(a) or np.iscomplexobj(b):raise TypeError('full multiplication requires real coefficients')
    a,b=np.broadcast_arrays(a.astype(np.float64),b.astype(np.float64))
    if stats is not None:stats.update(complex_nodes=0,real_root_nodes=0,real_bilinear_products_per_pair=0)
    if N==1:
        if stats is not None:stats['real_bilinear_products_per_pair']=1
        return a*b
    X,Y=a.copy(),b.copy();X[...,0]=Y[...,0]=0
    Z=real_cross(X,Y,stats)
    result=np.empty_like(a)
    result[...,0]=a[...,0]*b[...,0]-np.sum(X[...,1:]*Y[...,1:],axis=-1)
    result[...,1:]=a[...,0:1]*Y[...,1:]+b[...,0:1]*X[...,1:]+Z[...,1:]
    if stats is not None:stats['real_bilinear_products_per_pair']+=3*N-2
    return result


def arithmetic_counts(level):
    if level==0:return dict(M=1,A=0,S=0,T=1)
    if level<3:
        M,A,S=(4,2,0) if level==1 else (16,12,0)
    else:
        N=1<<level
        M=(9*level-15)*N//2+10
        A=(23*level-61)*N//2+36
        S=(2*level-7)*N//2+4
    return dict(M=M,A=A,S=S,T=M+A+S)


def direct(a,b):
    """Independent four-call recursion, for small correctness references only."""
    N=a.shape[-1]
    if N==1:return a*b
    h=N//2
    x,y=a[...,:h],a[...,h:];z,w=b[...,:h],b[...,h:]
    def conjugate(x):
        out=x.copy();out[...,1:]*=-1
        return out
    return np.concatenate((direct(x,z)-direct(conjugate(w),y),
                           direct(w,x)+direct(y,conjugate(z))),axis=-1)
