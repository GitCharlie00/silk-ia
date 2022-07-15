import cv2 as cv
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog as fd
from sklearn.preprocessing import StandardScaler

#################################################################### Image segmentation ###########################################################

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
    leaves_mask_space = cv.COLOR_RGBA2BGR

    lower = np.array([100, 150, 170], np.uint8)
    upper = np.array([255, 255, 255], np.uint8)
    
    pic = cv.cvtColor(src,leaves_mask_space)
    mask = cv.inRange(pic, lower, upper)
    res = cv.bitwise_and(pic,pic,mask = mask)  
    fin = cv.cvtColor(res,leaves_mask_space)
    
    return fin

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

### Prende le immagini e restituisce un array np con le features
def get_data(path):
    src = cv.imread(path)
    no_rete = remove_web(src)

    out_nobachi = remove_worms(no_rete)
    out_nofoglie = remove_leaves(no_rete)

    presenza_foglie = float(get_leaves_presence(out_nobachi))
    assenza_foglie = float(get_black(out_nobachi))

    presenza_bachi_sfondo = float(get_worms_presence(out_nofoglie))
    assenza_bachi_sfondo = float(get_black(out_nofoglie))
        
    out = [0, presenza_foglie,assenza_foglie,presenza_bachi_sfondo,assenza_bachi_sfondo]

    return out

################################################################ Logistic ################################################################

def logistic(z):
    den = 1 + np.exp(-z)
    sigm = 1.0/den
    return sigm

def hyp(W,X):
    param = np.dot(W,X.T)
    return logistic(param)

def prediction(W,X):
    h = hyp(W,X)
    Y_hat = h > 0.5
    return Y_hat

################################################################ App ################################################################

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.filetypes = (('Pictures', '*.jpg'),('All files', '*.*'))
        self.W = np.array(np.zeros((1,5)))
        self.X = np.array(np.zeros((1,5)))
        self.img = ImageTk.PhotoImage(Image.open("./Logo.jpg").resize((700,400)))

        self.grid_columnconfigure((0, 1, 2), weight=1)
        
        self.predict_but = tk.Button(self, text="Inserisci un immagine", font=('Helvetica', 12, "bold"), command=self.work)
        self.predict_but.grid(row=1,column=1,columnspan=3, pady=(15,15))
        
        self.show_res = tk.Label(self, image = self.img)
        self.show_res.grid(row=2,column=1)
        
        self.predict_label = tk.Label(self, text="Il tuo risultato apparirà qui", font=('Helvetica', 12, "bold"))
        self.predict_label.grid(row=3,column=1,columnspan = 3, pady=(15,0))
    
    def work(self):
        self.set_W()

        path = self.get_img()
        X_val = get_data(path)
        self.set_X(X_val)
        
        pred = prediction(self.W,self.X)

        if(pred):
            self.predict_label.config(text="Serve dar da mangiare ai bachi")
        else:
            self.predict_label.config(text="Non è necessario dar da mangiare ai bachi")

    def get_img(self):
        file_path = fd.askopenfilename(title='Inserisci un immagine', initialdir='../color_extr/sample', filetypes=self.filetypes)
        
        new_pic = ImageTk.PhotoImage(Image.open(file_path).resize((700,400)))
        self.show_res.configure(image=new_pic)
        self.show_res.image = new_pic
        
        path_rel_items = file_path.split("/")[11:]
        path = "../" + "/".join(path_rel_items)
        
        return path
    
    def set_W(self):
        self.W[0][0] =  0.00330611
        self.W[0][1] = -0.94125539
        self.W[0][2] = -0.00313383
        self.W[0][3] = 0.95150957
        self.W[0][4] = 0.00340949

    def set_X(self, val):
        self.X[0][0] = val[0]
        self.X[0][1] = val[1]
        self.X[0][2] = val[2]
        self.X[0][3] = val[3]
        self.X[0][4] = val[4]

################################################################ MAIN ################################################################
app = App()
app.title("TecnoSetaApp - Torcitoio")
app.geometry("800x550+250+100")
app.mainloop()