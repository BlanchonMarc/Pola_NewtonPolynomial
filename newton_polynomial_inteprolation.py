'''
Blanchon Marc - ERL VIBOT CNRS 6000 - 2019

This code is the python conversion of Ning Li's release matlab code

Please refer below for references 

% The code can only be used for research purpose.

% Please cite the following paper when you use it:
   % Ning Li, Yongqiang Zhao, Quan Pan, and Seong G. Kong, 
   % "Demosaicking DoFP images using Newton��s polynomial interpolation and polarization difference model" 
   % Optics Express 27, 1376-1391 (2019)

%Note:
%   The code is not optimized and may have bugs. There are ways to improve the efficiency of the algorithms. 
%   Your revision and improvement are welcome!
%   All the notes in this code correspond to the cases explained in the
%   original paper.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Newton Polynomial Interpolation                          %
%                                                          %
% Copyright (C) 2019 Ning Li. All rights reserved.         %
%                    ln_neo@mail.nwpu.edu.cn               %
%                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

'''

import numpy as np


def interpolate(I):
    I = np.double(I)
    (m, n) = I.shape

    R = np.zeros((m, n, 4))

    # Labeling different polatiration channels
    O = np.zeros((m, n), dtype=int)

    step = 2
    O[0:m+1:step, 0:n+1:step] = 0
    O[0:m+1:step, 1:n+1:step] = 1
    O[1:m+1:step, 1:n+1:step] = 2
    O[1:m+1:step, 0:n+1:step] = 3

    # Store intermediate results
    Y1 = I
    Y2 = I
    #Y1 = np.double(I)
    #Y2 = np.double(I)

    # for index in range(R.shape[2]):
    R[:, :, 0] = I
    R[:, :, 1] = I
    R[:, :, 2] = I
    R[:, :, 3] = I

    '''
    % Stage one interpolation: interpolate vertically for case Fig.6(b),
    % interpolate horizontally for case Fig.6(c), interpolate in diagonal
    % directions for case Fig.6(a). The Eqs.(14)-(17) are simplified in this
    % code.
    '''

    for i in range(3, m-4):
        for j in range(3, n-4):
            R[i, j, O[i, j]] = I[i, j]
            R[i, j, O[i, j+1]] = 0.5*I[i, j] + 0.0625*I[i, j-3] - 0.25*I[i, j-2] + \
                0.4375*I[i, j-1] + 0.4375*I[i, j+1] - \
                0.25*I[i, j+2] + 0.0625*I[i, j+3]
            R[i, j, O[i+1, j]] = 0.5*I[i, j] + 0.0625*I[i-3, j] - 0.25*I[i-2, j] + \
                0.4375*I[i-1, j] + 0.4375*I[i+1, j] - \
                0.25*I[i+2, j] + 0.0625*I[i+3, j]
            Y1[i, j] = 0.5*I[i, j] + 0.0625*I[i-3, j-3] - 0.25*I[i-2, j-2] + 0.4375 * \
                I[i-1, j-1] + 0.4375*I[i+1, j+1] - \
                0.25*I[i+2, j+2] + 0.0625*I[i+3, j+3]
            Y2[i, j] = 0.5*I[i, j] + 0.0625*I[i-3, j+3] - 0.25*I[i-2, j+2] + 0.4375 * \
                I[i-1, j+1] + 0.4375*I[i+1, j-1] - \
                0.25*I[i+2, j-2] + 0.0625*I[i+3, j-3]
    # One can adjust for better result
    thao = 5.8
    # Fusion of the estimations with edge classifier for case Fig.6(a).

    for i in range(3, m-4):
        for j in range(3, m-4):
            pha1 = 0.0
            pha2 = 0.0

            for k in range(-2, 3, 2):
                for l in range(-2, 3, 2):
                    pha1 = pha1 + abs(Y1[i+k, j+l] - I[i+k, j+l])
                    pha2 = pha2 + abs(Y2[i+k, j+l] - I[i+k, j+l])

            if (pha1 / pha2) > thao:
                R[i, j, O[i+1, j+1]] = Y2[i, j]
            elif (pha2/pha1) > thao:
                R[i, j, O[i+1, j+1]] = Y1[i, j]
            elif (((pha1/pha2) < thao) and ((pha2/pha1) < thao)):
                d1 = abs(I[i-1, j-1] - I[i+1, j+1]) + \
                    abs(2*I[i, j] - I[i-2, j-2] - I[i+2, j+2])
                d2 = abs(I[i+1, j-1] - I[i-1, j+1]) + \
                    abs(2*I[i, j] - I[i+2, j-2] - I[i-2, j+2])
                epsl = 0.000000000000001
                w1 = 1/(d1 + epsl)
                w2 = 1/(d2+epsl)
                R[i, j, O[i+1, j+1]] = (w1*Y1[i, j] + w2*Y2[i, j])/(w1 + w2)

    RR = R

    XX1 = I
    XX2 = I
    YY1 = I
    YY2 = I

    # Stage two interpolation: interpolate horizontally for case Fig.6(b),
    # interpolate vertically for case Fig.6(c).

    for i in range(3, m-4):
        for j in range(3, n-4):
            XX1[i, j] = R[i, j, O[i, j+1]]
            XX2[i, j] = 0.5*I[i, j] + 0.0625 * \
                R[i-3, j, O[i, j+1]] - 0.25*I[i-2, j]
            XX2[i, j] = XX2[i, j] + 0.4375 * \
                R[i-1, j, O[i, j+1]] + 0.4375*R[i+1, j, O[i, j+1]]
            XX2[i, j] = XX2[i, j] - 0.25*I[i+2, j] + 0.0625*R[i+3, j, O[i, j+1]]
            YY1[i, j] = R[i, j, O[i+1, j]]
            YY2[i, j] = 0.5*I[i, j] + 0.0625 * \
                R[i, j-3, O[i+1, j]] - 0.25*I[i, j-2]
            YY2[i, j] = YY2[i, j] + 0.4375 * \
                R[i, j-1, O[i+1, j]] + 0.4375*R[i, j+1, O[i+1, j]]
            YY2[i, j] = YY2[i, j] - 0.25*I[i, j+2] + 0.0625*R[i, j+3, O[i+1, j]]

    # Fusion of the estimations with edge classifier for case Fig.6(b) and Fig.6(c).

    for i in range(3, m-4):
        for j in range(3, n-4):
            pha1 = 0.0
            pha2 = 0.0

            for k in range(-2, 3, 2):
                for l in range(-2, 3, 2):
                    pha1 = pha1 + abs(XX1[i+k, j+l] - I[i+k, j+l])
                    pha2 = pha2 + abs(XX2[i+k, j+l] - I[i+k, j+l])

            if (pha1 / pha2) > thao:
                R[i, j, O[i, j+1]] = XX2[i, j]
            elif (pha2/pha1) > thao:
                R[i, j, O[i, j+1]] = XX1[i, j]
            elif (((pha1/pha2) < thao) and ((pha2/pha1) < thao)):
                d1 = abs(I[i, j-1] - I[i, j+1]) + \
                    abs(2*I[i, j] - I[i, j-2] - I[i, j+2])
                d2 = abs(I[i+1, j] - I[i-1, j]) + \
                    abs(2*I[i, j] - I[i+2, j] - I[i-2, j])
                epsl = 0.000000000000001
                w1 = 1/(d1 + epsl)
                w2 = 1/(d2 + epsl)
                R[i, j, O[i, j+1]] = (w1*XX1[i, j] + w2*XX2[i, j])/(w1 + w2)

            pha1 = 0.0
            pha2 = 0.0

            for k in range(-2, 3, 2):
                for l in range(-2, 3, 2):
                    pha1 = pha1 + abs(YY1[i+k, j+l] - I[i+k, j+l])
                    pha2 = pha2 + abs(YY2[i+k, j+l] - I[i+k, j+l])

            if (pha1 / pha2) > thao:
                R[i, j, O[i+1, j]] = YY2[i, j]
            elif (pha2/pha1) > thao:
                R[i, j, O[i+1, j]] = YY1[i, j]
            elif (((pha1/pha2) < thao) and ((pha2/pha1) < thao)):
                d1 = abs(I[i, j-1] - I[i, j+1]) + \
                    abs(2*I[i, j] - I[i, j-2] - I[i, j+2])
                d2 = abs(I[i+1, j] - I[i-1, j]) + \
                    abs(2*I[i, j] - I[i+2, j] - I[i-2, j])
                epsl = 0.000000000000001
                w1 = 1/(d1 + epsl)
                w2 = 1/(d2 + epsl)
                R[i, j, O[i, j+1]] = (w1*YY1[i, j] + w2*YY2[i, j])/(w1 + w2)

    R = RR
    I0 = R[:, :, 0]
    I45 = R[:, :, 1]
    I90 = R[:, :, 2]
    I135 = R[:, :, 3]

    return (I0, I45, I90, I135)
