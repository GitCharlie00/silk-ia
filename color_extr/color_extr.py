import cv2 as cv
import numpy as np

window_name = 'Originalie - No foglie - No bachi'
dim = (630, 550)

### Rimuove la retina verde
def rete_mask(pic):
    lower = np.array([51, 71, 21], np.uint8)
    upper = np.array([255, 255, 85], np.uint8)
    
    src = cv.imread(pic)
    mask = cv.inRange(src, lower, upper)
    mask_inv = cv.bitwise_not(mask)  
    res = cv.bitwise_and(src,src,mask = mask_inv)  
    
    return res

### Rimuove i bachi, mantiene foglie
def bachi_mask(pic):
    lower = np.array([110, 109, 109], np.uint8)
    upper = np.array([255, 255, 255], np.uint8)
    
    mask = cv.inRange(pic, lower, upper)
    mask_inv = cv.bitwise_not(mask)  
    res = cv.bitwise_and(pic,pic,mask = mask_inv)  
    
    return res

### Rimuove le foglie, mantiene fondo e bachi
def foglie_mask(pic):
    lower = np.array([0, 0, 0], np.uint8)
    upper = np.array([216, 207, 193], np.uint8)
    
    mask = cv.inRange(pic, lower, upper)
    mask_inv = cv.bitwise_not(mask)  
    res = cv.bitwise_and(pic,pic,mask = mask_inv)  
    
    return res

###################################################### MAIN ###################################################à

#for i in range(0,6):
#    no_rete = rete_mask('./sample/src'+str(i)+'.jpg')
#    out = bachi_mask(no_rete)
#    res_resized = cv.resize(out, dim, interpolation = cv.INTER_AREA)
#    cv.imshow(window_name, res_resized)
#    cv.waitKey(0)

for i in range(0,6):
    src = cv.imread('./sample/src'+str(i)+'.jpg')
    
    no_rete = rete_mask('./sample/src'+str(i)+'.jpg')
    out_nobachi = bachi_mask(no_rete)
    out_nofoglie = foglie_mask(no_rete)

    src_resized = cv.resize(src, dim, interpolation = cv.INTER_AREA)
    out_nobachi_resized = cv.resize(out_nobachi, dim, interpolation = cv.INTER_AREA)
    out_nofoglie_resized = cv.resize(out_nofoglie, dim, interpolation = cv.INTER_AREA)
    
    res = np.concatenate((src_resized,out_nofoglie_resized,out_nobachi_resized), axis=1)
    
    cv.imshow(window_name, res)
    cv.waitKey(0)