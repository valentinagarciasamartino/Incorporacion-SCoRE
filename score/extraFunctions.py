#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:01:41 2020

@author: emi
"""
#------ import packages --------- 
# import subprocess
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV,\
                                 Ridge 
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from scipy.signal import find_peaks
# from custom_scorer_module import custom_scoring_function
from sklearn.metrics import mean_squared_error,\
                            mean_absolute_error,\
                            make_scorer
from sklearn.model_selection import GroupKFold,\
                                    GroupShuffleSplit, ShuffleSplit
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#-------------------------------- 

def rigidbody_OCST(markers_track, markers_standing, num_cluster = '0', cluster_marker = ['none'], virtual_markers = ['none']):
    
    # -------------------------------------------------------
    if num_cluster == '0':
    
        numclústeres = int(input('Define the number of clústeres in your model:' ))
        print(f'Number of clústeres in your model is: {numclústeres}')
    else:
        numclústeres = int(num_cluster)
    
    aux = 1
    aux1 = numclústeres
    numbers_key = {}
    numbers_virtual_marker = {}
    while aux1 > 0:
        if cluster_marker[0] == 'none':
            # Define number of clústeres in model -----
            print(f'Select the mumbers of each marker for cluster number {aux}')
            
            for pos, key in enumerate(markers_track):
                print(f'{pos} -> {key}')
                
            # define markers of each cluster ---
            user_input = input('Enter Numbers\n').split(',')
        else:
            user_input = []
            for pos, key in enumerate(markers_track):
                for pos1, key_cluster in enumerate(cluster_marker[aux-1]):
                    if key == key_cluster:
                        user_input.append(str(pos))
                        print('cluster: ', aux, '-> marker: (', pos,':', key_cluster , ')')
                        # print(aux, pos, key_cluster)
                
        numbers_key['cluster_' + str(aux)] = [int(x.strip()) for x in user_input]
        
        if len(numbers_key['cluster_' + str(aux)]) < 3:
            print('ERROR: number of markers should be more than three.')
            break
        
        # Define virtual marked associated with each cluster ---
        if virtual_markers[0] == 'none':
            virtual_marker = 'n'
            virtual_marker = input(f'Do you have virtual markers associated with cluster number {aux} [y/N]:' )
            if virtual_marker.lower() == 'y':
                # Define number of clústeres in model -----
                print(f'Select the mumbers of each virtual marker in cluster number {aux}')
        
                pos = 0
                for key in markers_standing.keys():
                    print(f'{pos} -> {key}')
                    pos += 1
                    
                # define markers of each cluster ---
                user_input = input('Enter Numbers\n').split(',')
                numbers_virtual_marker['cluster_' + str(aux)] = [int(x.strip()) for x in user_input]
            else:
                numbers_virtual_marker['cluster_' + str(aux)] = []
        else:
            user_input = []
            for pos, key in enumerate(markers_standing):
                for pos1, key_virtual in enumerate(virtual_markers[aux-1]):
                    if key == key_virtual:
                        user_input.append(str(pos))
                        print('cluste: ', aux, '-> virtual marker: (', pos,':', key_virtual , ')')
            numbers_virtual_marker['cluster_' + str(aux)] = [int(x.strip()) for x in user_input]
        
        # New cluster ---
        aux += 1
        aux1 -= 1
        # end of while --
    
    # OCST: Optimal common shape technique ---------------------------------------
    # "The Procluster problem for rigid-body movement": soder function proposed by Soderkist y Per-Ake (1993)
    # Input:
    # x: 3-D marker coordinates in position 1 (3 columns, one row for each marker)
    # y: 3-D marker coordinates in position 2 (same format)
    # Output:
    # R: rotation matrix
    # d: translation vector
    #
    # the rigid body model is: y = R*x + d
    aux1 = numclústeres
    aux = 1
    new_markers_track = markers_track.copy()
    R_OCST = {}
    d_OCST = {}
    
    while aux1 > 0:
        x = []
        x1 = []
        x2 = []
        y = []
        y1 = []
        for i in range(0,len(numbers_key['cluster_' + str(aux)])):
            x1.append( markers_standing[list(markers_track.keys())[numbers_key['cluster_' + str(aux)][i]]] )
        x1 = np.asarray(x1)
        n , m , l = x1.shape
        x1 = x1.swapaxes(1,2)
        x = x1[:,int(np.fix(m/4)),:]
        
        for i in range(0,len(numbers_key['cluster_' + str(aux)])):
            y1.append( markers_track[list(markers_track.keys())[numbers_key['cluster_' + str(aux)][i]]] )
        y = np.asarray(y1)
        y = y.swapaxes(1,2)
                
        # SODER: ------------------
        # construct matrices A and B, substract the mean so there is only rotation
        mx = np.mean(x,0)
        my = np.mean(y,0)
                
        A = x - mx
        B = y - my
                
        nm , times , coord = B.shape
        R = np.zeros((coord,times,coord))
        d = np.zeros((times,coord))
        for frame in range(0,times):
            A1 = np.transpose(A)
            B1 = np.transpose(B[:,frame,:])
            # The singular value decomposition to calculate R with det(R)=1
            C = B1.dot(np.transpose(A1))
            P, T, Q = np.linalg.svd(C)
            Q = np.transpose(Q)
            R[:,frame,:] = P.dot(np.diag([1,1,np.linalg.det(P.dot(np.transpose(Q)))])).dot(np.transpose(Q))
            # Calculate the translation vector from the centroid of all markers
            d[frame,:] = np.transpose(my[frame,:]) - R[:,frame,:].dot(np.transpose(mx))
        
        # Save R and d ---------------------------
            
        R_OCST['cluster_' + str(aux)] = R
        d_OCST['cluster_' + str(aux)] = d
        
        # end SODER ----------------
            
        # REWRITE MARKERS DATA ------
        for i in range(0,len(numbers_key['cluster_' + str(aux)])):
            x2 = markers_standing[list(markers_track.keys())[numbers_key['cluster_' + str(aux)][i]]]
            x2 = np.asarray(x2)
            x2 = x2.swapaxes(0,1)
            n , m = x2.shape
            x_stand = x2[int(np.fix(m/4)),:]
            # new_markers_track[list(markers_track.keys())[numbers_key['cluster_' + str(aux)][i]]] = np.transpose( R.dot(np.transpose(x_stand)) + np. transpose(d) )
            new_markers_track[list(markers_track.keys())[numbers_key['cluster_' + str(aux)][i]]] = R.dot(np.transpose(x_stand)) + np. transpose(d)
        # Virtual markers associated to cluster
        for i in range(0,len(numbers_virtual_marker['cluster_' + str(aux)])):
            x2 = markers_standing[list(markers_standing.keys())[numbers_virtual_marker['cluster_' + str(aux)][i]]]
            x2 = np.asarray(x2)
            x2 = x2.swapaxes(0,1)
            n , m = x2.shape
            x_stand = x2[int(np.fix(m/4)),:]
            # new_markers_track[list(markers_standing.keys())[numbers_virtual_marker['cluster_' + str(aux)][i]]] = np.transpose( R.dot(np.transpose(x_stand)) + np.transpose(d) )
            new_markers_track[list(markers_track.keys())[numbers_key['cluster_' + str(aux)][i]]] = R.dot(np.transpose(x_stand)) + np. transpose(d)
            
        # New cluster ---
        aux += 1    
        aux1 -= 1
        # end of while --
        
    # end OCST ---------------------------------------------------------------
    return new_markers_track, R_OCST, d_OCST
    
def SCORE_SARA(R_OCST, d_OCST, numSegment = ['0']):    
    
    # -------------------------------------------------------------------------
    # SCoRE - SARA ------------------------------------------------------------
    # -------------------------------------------------------------------------    
    if numSegment[0] == '0':
        print('Define the cluster number of two segments linked by CoR:' )
        
        pos = 1
        for key in R_OCST.keys():
            print(f'{pos} -> {key}')
            pos += 1
            
        # define markers of each cluster ---
        numSegment = input('Enter Numbers\n').split(',')   
        
    else:
        print('The cluster linked with CoR are:')
        pos = 1
        for key in R_OCST.keys():
            print(f'{pos} -> {key}')
            pos += 1
    
    if len(numSegment) < 2:
        print('ERROR: number of clústeres should be two.')
    else:
        # SCoRE/SARA methods: -------------------------------------------------
        # R.M. Ehrig et al. / Journal of Biomechanics 39 (2006) 2798-2809
        # R.M. Ehrig et al. / Journal of Biomechanics 40 (2007) 2150-2157   
        
        if numSegment[0] < numSegment[1]:
            Rp1 = R_OCST[list(R_OCST.keys())[int(numSegment[0])-1]]
            Op1 = d_OCST[list(R_OCST.keys())[int(numSegment[0])-1]]
            Rd1 = R_OCST[list(R_OCST.keys())[int(numSegment[1])-1]]
            Od1 = d_OCST[list(R_OCST.keys())[int(numSegment[1])-1]]
        elif numSegment[0] > numSegment[1]:
            Rp1 = R_OCST[list(R_OCST.keys())[int(numSegment[1])-1]]
            Op1 = d_OCST[list(R_OCST.keys())[int(numSegment[1])-1]]
            Rd1 = R_OCST[list(R_OCST.keys())[int(numSegment[0])-1]]
            Od1 = d_OCST[list(R_OCST.keys())[int(numSegment[0])-1]]
        
        R_SCoRE = np.zeros((len(Rp1[1])*3,3))
        t_SCoRE = np.zeros(len(Rp1[1])*3)
        S_SCoRE = np.zeros((len(Rd1[1])*3,3))
        d_SCoRE = np.zeros(len(Rp1[1])*3)  
        
        SCoREctr = np.zeros(6)
        SARAax = np.zeros(6)
            
        for k in range(0,len(Rp1[1])):
            R_SCoRE[(k*3):(k*3)+3,:] = Rp1[:,k,:]
            t_SCoRE[(k*3):(k*3)+3] = Op1[k,:]
            S_SCoRE[(k*3):(k*3)+3,:] = Rd1[:,k,:]
            d_SCoRE[(k*3):(k*3)+3] = Od1[k,:]
            
        A = np.concatenate((R_SCoRE,-S_SCoRE), axis=1)
        b = d_SCoRE - t_SCoRE
        
        
        # solve the linear equation Ax=b using QR decomposition
        
        q , r = np.linalg.qr(A , mode='reduced')
        
        y = np.dot(q.T,b)
        x = np.dot(np.linalg.inv(r),y)
    
        SCoREctr = x
        
        # Relationship between the SCoRE residual and the error of the estimated CoR (Ehrig et al. (2011))
        r_s = (np.dot(A,x) - b)/((len(b))**(1/2))
        U, S, V = np.linalg.svd(A)
        err = (np.linalg.norm(r_s)/2 )+ (1 + (2/(np.sqrt(6)))*(np.sum(1./(S*S)))**(1/2))
               
        # the axes are calculated from the singular value decomposition of A
        
        U, S, V = np.linalg.svd(A)
        SARAax[0:3] = V[0:3,5]/np.linalg.norm(V[0:3,5])
        SARAax[3:6] = V[3:6,5]/np.linalg.norm(V[3:6,5])
    # end SCoRE and SARA ------------------------------------------------------
    return SCoREctr, SARAax, err
