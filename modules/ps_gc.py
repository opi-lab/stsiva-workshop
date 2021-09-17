from scipy import ndimage
import numpy as np
import cv2

def NStepPhaseShifting(imgs, N):
    delta = 2*np.pi*np.arange(1,N+1)/N

    sumIsin, sumIcos = 0., 0.
    for fname, deltak in zip(imgs, delta):
        I = cv2.imread(fname, 0)
        sumIsin += I*np.sin(deltak)
        sumIcos += I*np.cos(deltak)
    
    return -np.arctan2(sumIsin, sumIcos)

def codeword(imlist):
    l = np.array(imlist).reshape(-1,2)
    code_word = []
    for imn1, imn2 in l:
        im1 = cv2.imread(imn1,0)
        im2 = cv2.imread(imn2,0)
        code_word.append(im1 > im2)

    return np.dstack(code_word)

def grayToDec(gray):
    n = gray.shape[-1]
    dec = np.zeros(gray.shape[:2])
    
    tmp = gray[...,0]
    dec[tmp] += 2**(n-1)
    
    for i in range(1,n):
        tmp ^= gray[...,i]
        dec[tmp] += 2**(n-i-1)
    
    return dec

def phaseGraycodingUnwrap(imlist_ps, imlist_gc, p, N):
    # Estimate wrapped phase map
    phi = NStepPhaseShifting(imlist_ps, N)

    # Estimate code words
    code_word = codeword(imlist_gc)

    # Estimate fringe order with codeword
    k = grayToDec(code_word)
    
    # Shift and rewrap wrapped phase
    shift = -np.pi + np.pi/p
    phi = np.arctan2(np.sin(phi+shift), np.cos(phi+shift))

    # Estimate absolute phase map
    Phi = phi + 2*np.pi*k
    
    # Shift phase back to the original values
    Phi -= shift
    
    # Filter spiky noise
    Phim = ndimage.median_filter(Phi, 5)
    Phi -= 2*np.pi*np.round((Phi-Phim)/2/np.pi)

    return Phi