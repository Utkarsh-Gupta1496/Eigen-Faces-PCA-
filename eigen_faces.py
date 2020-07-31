"""
Created on Thu Dec  5 18:52:34 2019

@author: utkarsh
"""

import numpy as np
import  matplotlib.pyplot as plt
import os
  
# Reading Data
data_dir = "./data"
file_  = os.listdir(data_dir)
result  =[]
for img in file_:
    image_path = os.path.join(data_dir,img)
    a = plt.imread(image_path,0)
    result.append(a)

#creating Data Matrix(List)
data_list=[]   
for i in result:
    data_list.append(i.flatten())
    
#Creating Data Matrix(Array)
data_matrix=np.array(data_list)

def normalize_mean(img,mean):
    normal=np.zeros((img.shape))
    a,b=img.shape
    for i in range(b):
        normal[:,i]=img[:,i]-mean
    return normal

def high_dimmensional_cov(normal):
    a,b=normal.shape
    "20 is number of data points"
    checkmatrix=np.matmul(np.transpose(normal),normal)/b
    return checkmatrix
    
def cov(normal):
    a,b=normal.shape
    "20 is number of data points"
    checkmatrix=np.matmul(normal,np.transpose(normal))/b
    return checkmatrix
    
def eigen_value_vector(matrix):
    return np.linalg.eig(matrix)

def final_eigen_vector(a,b,normal):
    u=np.zeros([77760,165])
    for i in range(165):
        if(a[i]!=0):
            u[:,i]=np.matmul(normal,b[:,i])/np.sqrt(a[i]*165)
    return (a,u)
    
def eigen_sort(value,vector):
    idx = value.argsort()[::-1]   
    eigenValues = value[idx]
    eigenVectors = vector[:,idx]
    return (eigenValues,eigenVectors)

# Performing High Dimmensional PCA
mean_data=np.mean(data_matrix,axis=0)
data_matrix_transposed=np.transpose(data_matrix)
normalized_data_matrix=normalize_mean(data_matrix_transposed,mean_data)
high_covarinace=high_dimmensional_cov(normalized_data_matrix)
eig_val,eig_vector=eigen_value_vector(high_covarinace)
eig_val=np.round((np.absolute(eig_val)))
eig_valf,eig_vectorf=final_eigen_vector(eig_val,eig_vector,normalized_data_matrix)

#Final Sorted Eigen Values and Eigen Vectors
eig_vals,eig_vectors=eigen_sort(eig_valf,eig_vectorf)

eigen_faces=[]
#Selecting top k eigen values
k=8
for i in range(k):
    eigen_faces.append(eig_vectors[:,i].reshape(243,320))

#Ploting Eigen Faces
a=0
fig, axs = plt.subplots(2, int(k/2))
for i in range(2):
    for j in range(int(k/2)):
        axs[i,j].imshow(eigen_faces[a],cmap='gray')
        axs[i,j].set_title('EigenFace : % 2d'%(a+1))
        a=a+1
