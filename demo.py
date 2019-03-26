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

import numpy as np
import cv2
from newton_polynomial_inteprolation import interpolate as NPI

PATH = 'test.bmp'

if __name__ == '__main__':
    I = float(cv2.imread(PATH, -1))

    (P0, P45, P90, P135) = NPI(I)

    ii = 0.5*(PI0 + PI45 + PI90 + PI135)
    SOP = np.uint8(ii)
    q = PI0 - PI90
    u = PI45 - PI135
    dolp = sqrt(q**2 + u**2)
    Dolp = dolp/ii
    DOLPP = np.uint8(Dolp)
    aop = (1/2) * np.arctan2(u/q)
    AOPP = np.uint8(aop)

    cv2.imshow('Intensity', SOP)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('DOP', DOLPP)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('AOP', AOPP)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
