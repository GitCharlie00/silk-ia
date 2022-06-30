import cv2 as cv
import numpy as np

window_name = 'Originale - No foglie - No bachi'
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

        print("No Foglie= ")
        get_worms_presence(out_nofoglie)
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

    lower = np.array((48, 252, 3), dtype=np.uint8)
    upper = np.array((48, 252, 3), dtype=np.uint8)

    mask = cv.inRange(img, lower, upper)

    ratio_green = cv.countNonZero(mask)/(img.size/3)
    colorPercent = (ratio_green * 100)

    print('Presenza foglie:', np.round(colorPercent, 2))

### Estrae feature: presenza dei bachi (>20.0 dai da mangiare, altrimenti ok)
def get_worms_presence(img):
    b, g, r = cv.split(img)
    mask = (b != 0) & (g != 0) & (r != 0)
    img[mask] = (242, 255, 3)
    img[mask==0] = (0, 0, 0)

    lower = np.array((242, 255, 3), dtype=np.uint8)
    upper = np.array((242, 255, 3), dtype=np.uint8)

    mask = cv.inRange(img, lower, upper)

    ratio_green = cv.countNonZero(mask)/(img.size/3)
    colorPercent = (ratio_green * 100)

    print('Presenza bachi:', np.round(colorPercent, 2))

### Estrazione altre 2/3 features (percentuale nero (assenza foglie,bachi), sfondo,colore dominante source, colore recessivo source ?)

### Prende le immagini e restituisce un array np con le features
