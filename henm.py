import numpy as np
import sys

def NxN_from_3Nx3N(matrix):
    N3 = matrix.shape[0]
    N = N3//3
    k = np.empty((N,N),dtype=np.float64)
    for i in range(N):
        for j in range(N):
            k[i,j] = np.trace(matrix[i*3:i*3+3,j*3:j*3+3])
    return k    

def project_NxN_to_3Nx3N(hessian,dispVecs):
    N = hessian.shape[0]
    N3 = N*3
    newMat = np.zeros((N3,N3),dtype=np.float64)
    for i in range(N):
        for j in range(i):
            diff = dispVecs[i,j,:]
            newMat[i*3:i*3+3,j*3:j*3+3] = hessian[i,j] * np.outer(diff,diff)
            # symmetrize
            newMat[j*3:j*3+3,i*3:i*3+3] = newMat[i*3:i*3+3,j*3:j*3+3].T
    # finish Hessian
    for i in range(N):
        for j in range(N):
            if j != i:
                newMat[i*3:i*3+3,i*3:i*3+3] -= newMat[j*3:j*3+3,i*3:i*3+3]
    return newMat

def update_hessian(hessian,covarDiff,alpha):
    N = hessian.shape[0]
    for i in range(N-1):
        for j in range(i+1,N):
            hessian[i,j] += alpha * covarDiff[i,j]
            # make sure spring constants stay positive (hessian elements stay negative)
            if hessian[i,j] > 0.0: 
                hessian[i,j] = 0.0
                #print("positive value in Hessian")
            # symmetrize
            hessian[j,i] = hessian[i,j]
    # update diagonal values
    for i in range(N):
        hessian[i,i] = 0.0
        hessian[i,i] = -np.sum(hessian[i,:])
    return hessian

def update_hessian_cutoff(hessian,covarDiff,alpha,indeces):
    N = hessian.shape[0]
    for index in indeces:
        hessian[index] += alpha * covarDiff[index]
        # make sure spring constants stay positive (hessian elements stay negative)
        if hessian[index] > 0.0: 
            hessian[index] = 0.0
        # symmetrize
        hessian[index[::-1]] = hessian[index]
    # update diagonal values
    for i in range(N):
        hessian[i,i] = 0.0
        hessian[i,i] = -np.sum(hessian[i,:])
    return hessian

def compute_henm_diff(newCovar,targetCovar):
    N = newCovar.shape[0]
    diffMat = np.zeros(newCovar.shape)
    
    for i in range(N-1):
        for j in range(i+1,N):
            diffMat[i,j] = (newCovar[i,i] + newCovar[j,j] - 2*newCovar[i,j])**(-1) - (targetCovar[i,i] + targetCovar[j,j] - 2*targetCovar[i,j])**(-1)
            diffMat[j,i] = diffMat[i,j]
    return diffMat

def compute_henm_diff_tensor(newCovar,targetRes,dispVecs):
    N3 = newCovar.shape[0]
    N = N3//3
    diffMat = np.zeros((N,N),dtype=np.float64)
    
    for i in range(N-1):
        for j in range(i+1,N):
            temp = newCovar[i*3:i*3+3,i*3:i*3+3] + newCovar[j*3:j*3+3,j*3:j*3+3] - newCovar[i*3:i*3+3,j*3:j*3+3] - newCovar[i*3:i*3+3,j*3:j*3+3].T
            diff = dispVecs[i,j,:]
            diffMat[i,j] = 1.0 / np.dot(diff.T,np.dot(temp,diff)) - targetRes[i,j]
            diffMat[j,i] = diffMat[i,j]
    return diffMat

def residual_from_covar(covar,dispVecs):
    N3 = covar.shape[0]
    N = N3//3
    residual = np.zeros((N,N),dtype=np.float64)
    for i in range(N-1):
        for j in range(i+1,N):
            temp = covar[i*3:i*3+3,i*3:i*3+3] + covar[j*3:j*3+3,j*3:j*3+3] - covar[i*3:i*3+3,j*3:j*3+3] - covar[i*3:i*3+3,j*3:j*3+3].T
            diff = dispVecs[i,j]
            residual[i,j] = residual[j,i] = 1.0 / np.dot(diff.T,np.dot(temp,diff))
    return residual

# compute pairwise displacement vectors
def pairwise_disp_vecs(pos):
    N = pos.shape[0]
    dispVecs = np.empty((N,N,3),dtype=np.float64)
    for i in range(N-1):
        for j in range(i+1,N):
            diff = pos[i,:] - pos[j,:]
            diff /= np.linalg.norm(diff)
            dispVecs[i,j,:] = dispVecs[j,i,:] = diff
    return dispVecs
            
# compute pairwise displacement vectors
def pairwise_cutoff(pos,cutoff):
    N = pos.shape[0]
    cutoffIndeces = []
    for i in range(N-1):
        for j in range(i+1,N):
            diff = pos[i,:] - pos[j,:]
            dist = np.linalg.norm(diff)
            if dist < cutoff:
                cutoffIndeces.append((i,j))
    return cutoffIndeces
            

def perform_henm(targetCovar,avgPos,guessHess = [],alpha=0.1,maxIter=10, thresh=1e-4):
    N3 = targetCovar.shape[0]
    N = N3//3
    # guess a Hessian if none is passed
    if guessHess == []:
        # create initial Hessian
        hessian = -10.0*np.ones((N,N),dtype=np.float64)
        #hessian = reachHessian
        for i in range(N):
            hessian[i,i] = 0.0
            hessian[i,i] = -np.sum(hessian[i,:])
    else:
        hessian = guessHess
    # compute pairwise displacement vectors
    dispVecs = pairwise_disp_vecs(avgPos)
    # convert target covar to target residual
    targetRes = residual_from_covar(targetCovar,dispVecs)
    # iterate
    step = 0
    conv = "FALSE"
    while step < maxIter and conv == "FALSE":
        # project hessian in 3Nx3N
        hessian3N = project_NxN_to_3Nx3N(hessian,dispVecs)
        # invert hessian
        covar3N = np.linalg.pinv(hessian3N,rcond=1e-10) * 0.6
        # take difference of tensor covars and project into separation vector
        covarDiff = compute_henm_diff_tensor(covar3N,targetRes,dispVecs)
        # Check if conveged
        dev = np.linalg.norm(covarDiff)
        if dev < thresh:
            conv = "TRUE"
        else: # update Hessian
            hessian = update_hessian(hessian,covarDiff,alpha)
        step += 1
        print(step, dev)
        sys.stdout.flush()

    return hessian


def perform_henm_cutoff(targetCovar,avgPos,guessHess = [],alpha=0.1,maxIter=10, thresh=1e-4,cutoff=12.0):
    N3 = targetCovar.shape[0]
    N = N3//3
    # determine cutoff indeces
    indeces = pairwise_cutoff(avgPos,cutoff)
    # guess a Hessian if none is passed
    if guessHess == []:
        # create initial Hessian
        hessian = np.zeros((N,N),dtype=np.float64)
        for index in indeces:
            hessian[index] = -10.0
            # symmetrize
            hessian[index[::-1]] = -10.0
        # populate diagonal
        for i in range(N):
            hessian[i,i] = 0.0
            hessian[i,i] = -np.sum(hessian[i,:])
    else:
        hessian = guessHess
    # compute pairwise displacement vectors
    dispVecs = pairwise_disp_vecs(avgPos)
    # convert target covar to target residual
    targetRes = residual_from_covar(targetCovar,dispVecs)
    # iterate
    step = 0
    conv = "FALSE"
    while step < maxIter and conv == "FALSE":
        # project hessian in 3Nx3N
        hessian3N = project_NxN_to_3Nx3N(hessian,dispVecs)
        # invert hessian
        covar3N = np.linalg.pinv(hessian3N,rcond=1e-10) * 0.6
        # take difference of tensor covars and project into separation vector
        covarDiff = compute_henm_diff_tensor(covar3N,targetRes,dispVecs)
        # Check if conveged
        dev = np.linalg.norm(covarDiff)
        if dev < thresh:
            conv = "TRUE"
        else: # update Hessian
            hessian = update_hessian_cutoff(hessian,covarDiff,alpha,indeces)
        step += 1
        print(step, dev)
        sys.stdout.flush()

    return hessian
