import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy import ndimage
import math
import csv 

pixelReplace = 1
pourcentage = 3
taille_fenetre = 800
columlnSex = 4

id_patient = 1
Max_id_patient = 9
nom_fichier_csv = "C:/Users/Sanchez/Documents/JFR/SexDetectionDICOM/labels1.csv"





def nb_bone(matrice):
        etiquettes, nombre_de_formes = ndimage.label(matrice > 0.6)
        res = []
        for i in range(1, nombre_de_formes + 1):  
            pixels_forme = (etiquettes == i)
            coordonnees_pixels = np.argwhere(pixels_forme)
            if len(coordonnees_pixels) < 300 :
                continue

            centre_x = np.mean(coordonnees_pixels[:, 0])
            centre_y = np.mean(coordonnees_pixels[:, 1])
            
            etiquettes[int(centre_x)][int(centre_y)] = pixelReplace 

            #print(f"Centre de la forme {i} : ({centre_x}, {centre_y})")
            res.append((centre_x, centre_y))


        return res

# Function to take care of teh translation and windowing. 
def window_image(img, window_center,window_width, intercept, slope, rescale=True):
    img = (img*slope +intercept) #for translation adjustments given in the dicom file. 
    img_min = window_center - window_width//2 #minimum HU level
    img_max = window_center + window_width//2 #maximum HU level
    img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
    img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level
    if rescale: 
        img = (img - img_min) / (img_max - img_min)*255.0 
    return img
    
def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue: return int(x[0])
    else: return int(x)
    
def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def cancelNoise(data):
    Width = 600
    Level = 600
    
    image = data.pixel_array
    window_center, window_width, intercept, slope = get_windowing(data)
    
    output = window_image(image, Level, Width, intercept, slope, rescale = True)

    return output





for id_patient in range(0,Max_id_patient):
    try: 
        path = 'C:/Users/Sanchez/Documents/JFR/DATA/patient_' + str(id_patient)

        fichiers_dicom = [os.path.join(path, fichier) for fichier in os.listdir(path) if fichier.endswith('.dcm')]
        fichiers_dicom.sort(key=lambda x: pydicom.dcmread(x).InstanceNumber)

        nombre_de_fichiers = len(fichiers_dicom)
        nombre_a_afficher = int(nombre_de_fichiers * (pourcentage / 100))

        derniers_fichiers_dicom = []
        derniers_fichiers_dicom_brut = [] 
        for ind in range(0, len(fichiers_dicom[-nombre_a_afficher:]), 1):
            dernier_fichier = pydicom.dcmread(fichiers_dicom[-nombre_a_afficher:][ind])
            derniers_fichiers_dicom_brut.append(dernier_fichier)
            derniers_fichiers_dicom.append(dernier_fichier.pixel_array)


        derniers_fichiers_dicom_WT_Noise =  list(map(cancelNoise, derniers_fichiers_dicom_brut))

        Bone_Image = []

        femur_center = nb_bone(derniers_fichiers_dicom_WT_Noise[-1])
        for fichier in derniers_fichiers_dicom_WT_Noise:
            image_matrice = fichier
            new_image =  image_matrice
            for x,y in femur_center:
                for i in range(-5, 6):
                    try:
                        new_image[int(x + i)][int(y)] = pixelReplace 
                        new_image[int(x)][int(y + i)] = pixelReplace 
                    except IndexError:
                        print("index error")
                        

            x1,y1 = femur_center[0]
            x2,y2 = femur_center[1]

            x3,y3 = int((x1 + x2) /2), int((y1 + y2)/2 )
            for i in range(-10, 11):
                new_image[int((x1 + x2) /2) + i][int((y1 + y2)/2 )] = pixelReplace 
                new_image[int((x1 + x2) /2)][int((y1 + y2)/2 ) + i] = pixelReplace 
            Bone_Image.append(new_image)

        res = []
        for new_image in Bone_Image:
            dx = x2 - x1
            dy = y2 - y1
            d = -120
            norme = math.sqrt(dx**2 + dy**2)

            # Normalisation du vecteur 
            ux = dx / norme
            uy = dy / norme

            vx = -uy
            vy = ux

            vx *= d
            vy *= d

            x = x3 + vx
            y = y3 + vy

            for i in range(-10, 11):
                try:
                    new_image[int(x + i)][int(y)] = pixelReplace 
                    new_image[int(x)][int(y + i)] = pixelReplace 
                except IndexError :
                    print("erreur index")
            res.append((new_image,(x, y)))


        taille_f = 100 # pixels = taille_f **2 
        conteurH, conteurF = 0,0

        for ind in range(0,len(res)):
            image = derniers_fichiers_dicom[ind]
            final_image = image.astype("float32")
            final_image /= np.max(final_image)
            fenetre = final_image[int(x - taille_f // 2) : int(x + taille_f // 2 ),
                    int(y - taille_f // 2 ): int(y + taille_f // 2 )]

            pixels_blanc = np.sum(fenetre > 0.15)
            #print(pixels_blanc)
  
            if pixels_blanc > 4000 :
                conteurH +=1
            else:
                conteurF +=1

        print("patient : "  + str(id_patient)+ " H : " + str(conteurH) + " F : " + str(conteurF))
        
        nouveau_texte =  str(conteurH > conteurF)
        ligne_a_modifier = id_patient + 1  
        with open(nom_fichier_csv, 'r', newline='') as fichier_csv:
            lecteur_csv = list(csv.reader(fichier_csv))
            
            if ligne_a_modifier < len(lecteur_csv):
                ligne = lecteur_csv[ligne_a_modifier]
                
                colonne_a_modifier = 2 
                
                while len(ligne) <= colonne_a_modifier:
                    ligne.append("") 
                
                ligne[colonne_a_modifier] = nouveau_texte

        with open(nom_fichier_csv, 'w', newline='') as fichier_csv:
            ecrivain_csv = csv.writer(fichier_csv)
            ecrivain_csv.writerows(lecteur_csv)
    except ValueError:
        print("Erreur")
