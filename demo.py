'''
Blanchon Marc - ERL VIBOT CNRS 6000 - 2019

Transcription of Ning Li's matlab code

see information below

% The code can only be used for research purpose.

% Please cite the following paper when you use it:
   % Ning Li, Yongqiang Zhao, Quan Pan, and Seong G. Kong,
   % "Demosaicking DoFP images using Newton��s polynomial interpolation and polarization difference model"
   % Optics Express 27, 1376-1391 (2019)

%Note:
%   The code is not optimized and may have bugs. There are ways to improve the efficiency of the algorithms.
%   Your revision and improvement are welcome!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Newton Polynomial Interpolation                          %
%                                                          %
% Copyright (C) 2019 Ning Li. All rights reserved.         %
%                    ln_neo@mail.nwpu.edu.cn               %
%                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
from scipy.misc import bytescale
import numpy as np
import cv2
from newton_polynomial_inteprolation import interpolate as NPI

PATH = '/Users/marc/Documents/Dataset/AquisitionImagesFinal/Polarcam/image_00001.png'

if __name__ == '__main__':
    I = np.double(cv2.imread(PATH, 0))

    (PI0, PI45, PI90, PI135) = NPI(I)

    ii = 0.5*(PI0 + PI45 + PI90 + PI135)
    # print(bytescale(ii))
    #print(ii[0:10, 0:10])

    SOP = np.double(ii)
    #print(SOP[0:10, 0:10])
    out = np.zeros(SOP.shape, np.double)
    SOP = bytescale(cv2.normalize(SOP, out, 1.0, 0.0,
                                  cv2.NORM_MINMAX, dtype=cv2.CV_32F))
    #print(SOP[0:10, 0:10])
    # print(bytescale(SOP))
    # print(np.uint8(SOP[0:10, 0:10])PATH = '/Users/marc/Documents/Dataset/AquisitionImagesFinal/Polarcam/image_00001.png')
    # exit()
    q = PI0 - PI90
    u = PI45 - PI135
    dolp = np.sqrt(q*q + u*u)
    Dolp = dolp/ii
    #print(Dolp[1:10, 1:10])
    DOLPP = np.double(Dolp)
    #print(DOLPP[1:10, 1:10])
    out = np.zeros(DOLPP.shape, np.double)
    DOLPP = bytescale(cv2.normalize(DOLPP, out, 1.0, 0.0,
                                    cv2.NORM_MINMAX, dtype=cv2.CV_32F))
    #print(DOLPP[1:10, 1:10])
    aop = (1/2) * np.arctan2(u, q)

    AOPP = np.double(aop)

    out = np.zeros(AOPP.shape, np.double)
    AOPP = bytescale(cv2.normalize(AOPP, out, 1.0, 0.0,
                                   cv2.NORM_MINMAX, dtype=cv2.CV_32F))

    #SOP = np.uint8(SOP)
    #DOLPP = np.uint8(DOLPP)
    #AOPP = np.uint8(AOPP)

    cv2.imshow('Intensity', SOP)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('DOP', DOLPP)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('AOP', AOPP)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
