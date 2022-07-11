import cv2 as cv
import numpy as np
from os import listdir
import csv

window_name = 'Originale - No foglie - No bachi'
test_dir = "./sample/"
data_dir = "E:\\SilkIA\\100CANON\\"
dim = (430, 250)

### Rimuove la rete del letto dei bachi
def remove_web(pic):
    lower = np.array([51, 71, 21], np.uint8)
    upper = np.array([255, 255, 85], np.uint8)
    
    mask = cv.inRange(pic, lower, upper)
    mask_inv = cv.bitwise_not(mask)  
    res = cv.bitwise_and(pic,pic,mask = mask_inv)  

    cv.imshow(window_name, cv.resize(res, dim, interpolation = cv.INTER_AREA))
    cv.imwrite('./notebook_pic/no_web.jpg', res)
    cv.waitKey(0)

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
    leaves_mask_space = cv.COLOR_RGBA2BGR

    lower = np.array([100, 150, 170], np.uint8)
    upper = np.array([255, 255, 255], np.uint8)
    
    # Converto in BGR per via del maggior contrasto tra foglie, bachi e sfondo
    pic = cv.cvtColor(src,leaves_mask_space)
    mask = cv.inRange(pic, lower, upper)
    res = cv.bitwise_and(pic,pic,mask = mask)  
    fin = cv.cvtColor(res,leaves_mask_space)
    
    return fin

### Estrae feature: presenza delle foglie (<50.0 dai da mangiare, altrimenti ok)
def get_leaves_presence(img):
    # Ottengo i canali dell'immagine
    b, g, r = cv.split(img)
    # Creo una maschera che individuerà solo i pixel non neri (foglie)
    mask = (b != 0) & (g != 0) & (r != 0)
    # Applico la maschera alla foto sostituendo i pixel non neri con un unico colore (per poterli rilevare meglio)
    img[mask] = (48, 252, 3)
    img[mask==0] = (0, 0, 0)

    # Isolo i pixel rimasti
    lower = np.array((48, 252, 3), dtype=np.uint8)
    upper = np.array((48, 252, 3), dtype=np.uint8)

    mask = cv.inRange(img, lower, upper)

    # Calcolo la percentuale considerando i pixel non nulli
    ratio_green = cv.countNonZero(mask)/(img.size/3)
    colorPercent = (ratio_green * 100)

    return np.round(colorPercent, 2)

### Estrae feature: presenza dei bachi (>15.5 dai da mangiare, altrimenti ok)
def get_worms_presence(img):    
    b, g, r = cv.split(img)
    mask = (b != 0) & (g != 0) & (r != 0)
    img[mask] = (242, 255, 3)
    img[mask==0] = (0, 0, 0)

    lower = np.array((242, 255, 3), dtype=np.uint8)
    upper = np.array((242, 255, 3), dtype=np.uint8)

    mask = cv.inRange(img, lower, upper)

    ratio_blu = cv.countNonZero(mask)/(img.size/3)
    colorPercent = (ratio_blu * 100)

    return np.round(colorPercent, 2)

### Estrae feature: assenza di bachi/foglie
def get_black(img):
    b, g, r = cv.split(img)
    mask = (b == 0) & (g == 0) & (r == 0)
    img[mask] = (211, 52, 235)
    img[mask==0] = (0, 0, 0)

    lower = np.array((211, 52, 235), dtype=np.uint8)
    upper = np.array((211, 52, 235), dtype=np.uint8)

    mask = cv.inRange(img, lower, upper)

    ratio_yellow = cv.countNonZero(mask)/(img.size/3)
    colorPercent = (ratio_yellow * 100)

    return np.round(colorPercent, 2)

### Mostra i risultati dei vari filtri/maschere applicate alle foto
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
        
        # Computazione presenza colori
        print("Presenza foglie: ",get_leaves_presence(out_nobachi))
        print("Assenza foglie: ",get_black(out_nobachi))

        print("Presenza bachi e sfondo: ",get_worms_presence(out_nofoglie))
        print("Assenza bachi e sfondo: ",get_black(out_nofoglie))
        
        # Mostro il risultato
        cv.imshow(window_name, res)
        cv.waitKey(0)

### Prende le immagini e restituisce un array np con le features
def get_data():
    # Percorso cartella con le foto   
    folder_dir = data_dir
    # Numero totale di foto
    total_photoes = len(listdir(folder_dir))
    # Inizializzo l'array numpy (total_photoes, 5) che conterrà le 5 features (presenza/assenza foglie, presenza/assenza bachi-sfondo, classificazione)
    out = np.zeros(shape=(len(listdir(folder_dir)),5), dtype= 'object')
    # Definisco il tipo dell'ultima feature come intero
    out[:,1].astype(int)
    # Contatore per popolare l'array
    cnt = 0

    # Per ogni immagine
    for images in listdir(folder_dir):
        # Acquisisco l'immagine
        src = cv.imread(folder_dir+images)
        # Rimuovo la rete
        no_rete = remove_web(src)

        # Ottengo la foto senza rete ne bachi
        out_nobachi = remove_worms(no_rete)
        # Ottengo la foto senza rete ne foglie
        out_nofoglie = remove_leaves(no_rete)

        # Calcolo la presenza di foglie nella foto
        presenza_foglie = float(get_leaves_presence(out_nobachi))
        # Calcolo l'assenza di foglie nella foto
        assenza_foglie = float(get_black(out_nobachi))

        # Calcolo la presenza di di bachi-sfondo nella foto
        presenza_bachi_sfondo = float(get_worms_presence(out_nofoglie))
        # Calcolo l'assenza di bachi-sfondo nella foto
        assenza_bachi_sfondo = float(get_black(out_nofoglie))
        
        # Etichetto automaticamente i risultati con i due valori della classificazione
        if(presenza_foglie <= 50.0 and assenza_foglie >= 50.0 and presenza_bachi_sfondo >= 15.5 and assenza_bachi_sfondo <= 84.5): #dai da mangiare
            classificazione = 1
        else:   #non dare da mangiare
            classificazione = 0

        # Aggiungo i dati della foto nell'output
        out[[cnt]] = [[presenza_foglie,assenza_foglie,presenza_bachi_sfondo,assenza_bachi_sfondo,classificazione]]
        cnt+=1

    # Salvo il dataset creato
    save_data(out, listdir(folder_dir))

### Crea il dataset
def save_data(data, images):
    # Apro (creo) il file che conterrà il dataset
    f = open('./data.csv', 'w', newline='')
    # Inizializzo il writer
    writer = csv.writer(f)

    # Scrivo l'header (campo foto serve per test)
    writer.writerow(['presenza_foglie','assenza_foglie','presenza_bachi_sfondo','assenza_bachi_sfondo','classificazione','foto'])

    # Scrivo il dataset
    for i in range(0, len(data)):
        writer.writerow(np.append(data[i],images[i]))
    
    # Chiudo il file
    f.close()