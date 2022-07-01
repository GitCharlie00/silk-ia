import cv2 as cv
import numpy as np
from os import listdir

window_name = 'Originale - No foglie - No bachi'
test_dir = "./sample/"
data_dir = "E:\\SilkIA\\"
dim = (430, 250)
leaves_mask_space = cv.COLOR_RGBA2BGR 

### Rimuove la rete del letto dei bachi
def remove_web(pic):
    lower = np.array([51, 71, 21], np.uint8)
    upper = np.array([255, 255, 85], np.uint8)
    
    mask = cv.inRange(pic, lower, upper)
    mask_inv = cv.bitwise_not(mask)  
    res = cv.bitwise_and(pic,pic,mask = mask_inv)  
    
    return res

### Rimuove i bachi, mantiene foglie
def remove_worms(pic):
    lower = np.array([110, 109, 109], np.uint8)
    upper = np.array([255, 255, 255], np.uint8)
    
    mask = cv.inRange(pic, lower, upper)
    mask_inv = cv.bitwise_not(mask)  
    res = cv.bitwise_and(pic,pic,mask = mask_inv)  
    
    return res
### Rimuove le foglie, mantiene bachi
def remove_leaves(src):
    lower = np.array([100, 150, 170], np.uint8)
    upper = np.array([255, 255, 255], np.uint8)
    
    # Converto in BGR per via del maggior contrasto tra foglie, bachi e sfondo
    pic = cv.cvtColor(src,leaves_mask_space)
    mask = cv.inRange(pic, lower, upper)
    res = cv.bitwise_and(pic,pic,mask = mask)  
    fin = cv.cvtColor(res,leaves_mask_space)
    
    return fin

def show_samples():
    for i in range(0,6):
        # Carico una foto
        src = cv.imread('./sample/src'+str(i)+'.jpg')

        # Rimuovo la rete
        no_rete = remove_web(src)

        # Ottengo la foto senza rete ne bachi
        out_nobachi = remove_worms(no_rete)

        # Ottengo la foto senza rete ne foglie
        out_nofoglie = remove_leaves(no_rete)
        
        # Ridimensiono le immagini per check di qualità operazione
        src_resized = cv.resize(src, dim, interpolation = cv.INTER_AREA)
        out_nobachi_resized = cv.resize(out_nobachi, dim, interpolation = cv.INTER_AREA)
        out_nofoglie_resized = cv.resize(out_nofoglie, dim, interpolation = cv.INTER_AREA)
        
        # Posiziono le immagini in una sola finestra
        res = np.concatenate((src_resized,out_nofoglie_resized,out_nobachi_resized), axis=1)
        
        print("#############################################################################################")
        print("No Bachi= ")
        get_leaves_presence(out_nobachi)
        get_black(out_nobachi)

        print("No Foglie= ")
        get_worms_presence(out_nofoglie)
        get_black(out_nofoglie)

        print("#############################################################################################")
        
        # Mostro il risultato
        cv.imshow(window_name, res)
        cv.waitKey(0)

        

### Estrae feature: presenza delle foglie (<50.0 dai da mangiare, altrimenti ok)
def get_leaves_presence(img):
    b, g, r = cv.split(img)
    mask = (b != 0) & (g != 0) & (r != 0)
    img[mask] = (48, 252, 3)
    img[mask==0] = (0, 0, 0)

    #cv.imshow(window_name, cv.resize(img, dim, interpolation = cv.INTER_AREA))
    #cv.waitKey(0)

    lower = np.array((48, 252, 3), dtype=np.uint8)
    upper = np.array((48, 252, 3), dtype=np.uint8)

    mask = cv.inRange(img, lower, upper)

    ratio_green = cv.countNonZero(mask)/(img.size/3)
    colorPercent = (ratio_green * 100)

    #print('Presenza foglie:', np.round(colorPercent, 2))
    return np.round(colorPercent, 2)

### Estrae feature: presenza dei bachi (>20.0 dai da mangiare, altrimenti ok)
def get_worms_presence(img):
    b, g, r = cv.split(img)
    mask = (b != 0) & (g != 0) & (r != 0)
    img[mask] = (242, 255, 3)
    img[mask==0] = (0, 0, 0)

    #cv.imshow(window_name, cv.resize(img, dim, interpolation = cv.INTER_AREA))
    #cv.waitKey(0)

    lower = np.array((242, 255, 3), dtype=np.uint8)
    upper = np.array((242, 255, 3), dtype=np.uint8)

    mask = cv.inRange(img, lower, upper)

    ratio_blu = cv.countNonZero(mask)/(img.size/3)
    colorPercent = (ratio_blu * 100)

    #print('Presenza bachi e fondo:', np.round(colorPercent, 2))
    return np.round(colorPercent, 2)

### Estrae feature: assenza di bachi/foglie
def get_black(img):
    b, g, r = cv.split(img)
    mask = (b == 0) & (g == 0) & (r == 0)
    img[mask] = (211, 52, 235)
    img[mask==0] = (0, 0, 0)

    #cv.imshow(window_name, cv.resize(img, dim, interpolation = cv.INTER_AREA))
    #cv.waitKey(0)

    lower = np.array((211, 52, 235), dtype=np.uint8)
    upper = np.array((211, 52, 235), dtype=np.uint8)

    mask = cv.inRange(img, lower, upper)

    ratio_yellow = cv.countNonZero(mask)/(img.size/3)
    colorPercent = (ratio_yellow * 100)

    #print('Assenza di elementi:', np.round(colorPercent, 2))
    return np.round(colorPercent, 2)

  
### Prende le immagini e restituisce un array np con le features
def get_data():   
    # get the path/directory
    #folder_dir = "E:\\100CANON"
    folder_dir = test_dir
    out = np.zeros(shape=(len(listdir(folder_dir)),5))
    cnt = 0

    for images in listdir(folder_dir):
        src = cv.imread(folder_dir+images)
        no_rete = remove_web(src)

        out_nobachi = remove_worms(no_rete)
        out_nofoglie = remove_leaves(no_rete)

        presenza_foglie = float(get_leaves_presence(out_nobachi))
        assenza_foglie = float(get_black(out_nobachi))

        presenza_bachi_sfondo = float(get_worms_presence(out_nofoglie))
        assenza_bachi_sfondo = float(get_black(out_nofoglie))

        if(presenza_foglie <= 50.0 and assenza_foglie >= 50.0 and presenza_bachi_sfondo >= 20.0 and assenza_bachi_sfondo <= 80.0): #dai da mangiare
            classificazione = 1
        else:   #non dare da mangiare
            classificazione = 0

        out[[cnt]] = [[presenza_foglie,assenza_foglie,presenza_bachi_sfondo,assenza_bachi_sfondo,classificazione]]
        cnt+=1

    return out

get_data()