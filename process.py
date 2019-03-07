import cv2
from PIL import Image
import numpy as np


def process_interpolation(image):
    # print(image.shape)
    threshold = 5.7

    polarized0 = np.zeros_like(image)
    polarized45 = np.zeros_like(image)
    polarized90 = np.zeros_like(image)
    polarized135 = np.zeros_like(image)

    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            # print(x)
            I45_sum = 0
            I_45_sum = 0
            I45 = 0
            I_45 = 0
            I45_save = 0
            I_45_save = 0
            for indx in [x-2, x, x+2]:
                for indy in [y-2, y, y+2]:
                    # print(indx)
                    # print(indy)
                    if indx-1 >= 0 and indx+1 <= image.shape[0]-1 and indy-4 >= 0 and indy+2 <= image.shape[1]-1:
                        try:
                            I45 = np.uint16(image[indx, indy])
                            I45 += (image[indx+1, indy-1] +
                                    image[indx-1, indy+1]) / 2

                            inter1 = (image[indx+1, indy-1] +
                                      image[indx+1, indy-3])/2
                            inter1 -= (image[indx+1, indy] -
                                       (2*image[indx+1, indy-2])+image[indx+1, indy-4])/8

                            inter2 = (image[indx-1, indy+1] +
                                      image[indx-1, indy-1])/2
                            inter2 -= (image[indx-1, indy + 2] -
                                       (2*image[indx-1, indy])+image[indx-1, indy-2])/8

                            I45 -= (inter1 + inter2) / 2

                            # print(I45)

                            I_45 = np.uint16(image[indx, indy])
                            I_45 += (image[indx-1, indy-1] +
                                     image[indx+1, indy+1]) / 2

                            inter1 = (image[indx-1, indy-1] +
                                      image[indx-1, indy-3]) / 2
                            inter1 -= (image[indx-1, indy] -
                                       (2*image[indx-1, indy-2]) + image[indx-1, indy-4])/8

                            inter2 = (image[indx+1, indy+1] +
                                      image[indx+1, indy-1])/2
                            inter2 -= (image[indx+1, indy+2] -
                                       (2*image[indx+1, indy]) + image[indx+1, indy-2])/8

                            I_45 -= (inter1 + inter2) / 2
                            if indx == x and indy == y:
                                I45_save = I45
                                I_45_save = I_45
                            I45_sum += abs(I45 - image[indx, indy])
                            I_45_sum += abs(I_45 - image[indx, indy])

                            # print(I_45)
                        except ValueError:
                            pass

            # print(I45_sum)
            # print(I_45_sum)
            if I45_sum == 0 or I_45_sum == 0:
                vals = [0, 0]
            else:
                vals = [I45_sum/I_45_sum, I_45_sum/I45_sum]
            argmax = max(vals)
            index = vals.index(max(vals))
            # print(argmax)
            # print(index)
            val = 0
            if argmax > threshold and index == 1:
                val = I45
            elif argmax > threshold and index == 0:
                val = I_45_sum
            else:
                eps = 1e-12

                xvec = [x-2, x-1, x+1, x+2]
                yvec = [y-2, y-1, y+1, y+2]
                # print(xvec)
                # print(yvec)

                for indice, xvecnew in enumerate(xvec):
                    if xvecnew > image.shape[0]-1 or xvecnew < 0:
                        xvec[indice] = x

                for indice, yvecnew in enumerate(yvec):
                    if yvecnew > image.shape[1]-1 or yvecnew < 0:
                        yvec[indice] = y

                d45 = np.uint16(abs(image[xvec[2], yvec[1]] - image[xvec[1], yvec[2]]) + abs(
                    2*image[x, y] - image[xvec[0], yvec[3]] - image[xvec[3], yvec[0]]))
                d_45 = np.uint16(abs(image[xvec[1], yvec[1]]-image[xvec[2], yvec[2]]) + abs(
                    2*image[x, y] - image[xvec[0], yvec[0]] - image[xvec[3], yvec[3]]))

                om45 = 1 / (d45 + eps)
                om_45 = 1 / (d_45 + eps)
                val = ((om45 * I45_save) + (om_45*I_45_save))/(om45 + om_45)
            if x % 2 == 0:
                if y % 2 == 0:
                    polarized90[x, y] = val

                else:
                    polarized45[x, y] = val
            else:
                if y % 2 == 0:
                    polarized135[x, y] = val
                else:
                    polarized0[x, y] = val

    return [polarized0, polarized45, polarized135, polarized90]


def fill_values(i0, i45, i135, i90, original):

    for indx in range(original.shape[0]):
        for indy in range(original.shape[1]):

            if indx % 2 == 0:

                if indy % 2 == 0:
                    i0[indx, indy] = original[indx, indy]
                else:
                    i135[indx, indy] = original[indx, indy]

            else:

                if indy % 2 == 0:
                    i45[indx, indy] = original[indx, indy]
                else:
                    i90[indx, indy] = original[indx, indy]
    return [i0, i45, i135, i90]


def interpolate_pol(i0, i45, i135, i90):

    # inter1 = Image.fromarray(i0)
    # img = inter1.resize((int(640 / 2), int(480 / 2)), Image.ANTIALIAS)
    # src = i0
    # img = cv2.resize(src, (int(640 / 2), int(480 / 2)),
    #                 interpolation=cv2.INTER_LINEAR)
    # inter2 = img
    # img = cv2.resize(inter2, (int(640), int(480)),
    #                 interpolation=cv2.INTER_LINEAR)

    # i0 = np.array(img)

    # inter1=Image.fromarray(i45)
    # img=inter1.resize((int(640 / 2), int(480 / 2)), Image.ANTIALIAS)
    # inter2=img
    # img=inter2.resize((int(640), int(480)), Image.ANTIALIAS)

    # i45=np.array(img)

    # src = i45
    # img = cv2.resize(src, (int(640 / 2), int(480 / 2)),
    #                 interpolation=cv2.INTER_LINEAR)
    # inter2 = img
    # img = cv2.resize(inter2, (int(640), int(480)),
    #                 interpolation=cv2.INTER_LINEAR)

    # i45 = np.array(img)

    # src = i135
    # img = cv2.resize(src, (int(640 / 2), int(480 / 2)),
    #                 interpolation=cv2.INTER_LINEAR)
    # inter2 = img
    # img = cv2.resize(inter2, (int(640), int(480)),
    #                 interpolation=cv2.INTER_LINEAR)

    # i135 = np.array(img)

    # src = i90
    # img = cv2.resize(src, (int(640 / 2), int(480 / 2)),
    #                 interpolation=cv2.INTER_LINEAR)
    # inter2 = img
    # img = cv2.resize(inter2, (int(640), int(480)),
    #                 interpolation=cv2.INTER_LINEAR)

    # i90 = np.array(img)
    # inter1=Image.fromarray(i135)
    # img=inter1.resize((int(640 / 2), int(480 / 2)), Image.ANTIALIAS)
    # inter2=img
    # img=inter2.resize((int(640), int(480)), Image.ANTIALIAS)

    # i135=np.array(img)

    # inter1=Image.fromarray(i90)
    # img=inter1.resize((int(640 / 2), int(480 / 2)), Image.ANTIALIAS)
    # inter2=img
    # img=inter2.resize((int(640), int(480)), Image.ANTIALIAS)

    # i90=np.array(img)

    for indx in range(1, i0.shape[0]-1):
        for indy in range(1, i0.shape[1]-1):

            if (indx % 2 == 0 and indy % 2 == 0) or (indx % 2 != 0 and indy % 2 != 0):
                # 135 , 45
                i135[indx, indy] = (
                    i135[indx-1, indy] + i135[indx+1, indy] + i135[indx, indy-1] + i135[indx, indy+1])/4
                i45[indx, indy] = (
                    i45[indx-1, indy] + i45[indx+1, indy] + i45[indx, indy-1] + i45[indx, indy+1])/4
            elif (indx % 2 == 0 and indy % 2 != 0) or (indx % 2 != 0 and indy % 2 == 0):
                # 00 , 90
                i0[indx, indy] = (i0[indx-1, indy] + i0[indx+1,
                                                        indy] + i0[indx, indy-1] + i0[indx, indy+1])/4
                i90[indx, indy] = (
                    i90[indx-1, indy] + i90[indx+1, indy] + i90[indx, indy-1] + i90[indx, indy+1])/4
            # elif indx % 2 != 0 and indy % 2 == 0:
                # 0 , 90
            # elif indx % 2 != 0 and indy % 2 != 0:
                # 135 , 45

    return [i0, i45, i135, i90]


def convert_to_HSL(i0, i45, i135, i90):

    # print(np.amax(i0))
    # print(np.amax(i45))
    # print(np.amax(i90))
    # print(np.amax(i135))
    i0 = np.uint8(i0)
    i45 = np.uint8(i45)
    i90 = np.uint8(i90)
    i135 = np.uint8(i135)

    Js = [i0, i45, i90, i135]

    inten = (Js[0]+Js[1]+Js[2]+Js[3])/2.
    aop = (0.5*np.arctan2(Js[1]-Js[3], Js[0]-Js[2]))
    dop = np.sqrt((Js[1]-Js[3])**2+(Js[0]-Js[2])**2) / \
        (Js[0]+Js[1]+Js[2]+Js[3]+np.finfo(float).eps)*2
    cv2.imwrite(
        "/Users/marc/Github/Pola_NewtonPolynomial/output/inten.tiff", np.uint16(inten))
    cv2.imwrite(
        "/Users/marc/Github/Pola_NewtonPolynomial/output/aop.tiff", np.uint16(aop))
    cv2.imwrite(
        "/Users/marc/Github/Pola_NewtonPolynomial/output/dop.tiff", np.uint16(dop))
    #inten = (i0+i45+i90+i135)/2.
    #aop = (0.5*np.arctan2(i45-i135, i0-i45))
    #dop = np.sqrt((i45-i135)**2+(i0-i45)**2) / (i0 + i90 + np.finfo(float).eps)
    aopt = np.float64(2*((aop+np.pi/2)/np.pi*180))
    print(aopt)
    print(aopt.dtype)
    dopt = (dop/dop.max())*255
    print(dopt)
    print(dopt.dtype)
    dopa = dop*255
    print(dopa)
    print(dopa.dtype)
    intet = inten/inten.max()*255
    print(intet)
    print(intet.dtype)
    # cv2.imwrite(
    #    "/Users/marc/Github/Pola_NewtonPolynomial/output/inten.tiff", np.uint16(inten))
    # cv2.imwrite(
    #    "/Users/marc/Github/Pola_NewtonPolynomial/output/aop.tiff", np.uint16(aop))
    # cv2.imwrite(
    #    "/Users/marc/Github/Pola_NewtonPolynomial/output/dop.tiff", np.uint16(dop))

    hsv = np.zeros((i0.shape[0], i0.shape[1], 3))

    #hsv[:, :, 0] = (aop+np.pi/2)/np.pi*180
    #hsv[:, :, 1] = dop*255
    #hsv[:, :, 2] = inten/inten.max()*255
    #hsv = np.uint8(hsv)
    hsv = np.uint8(cv2.merge((np.float64((aop+np.pi/2)/np.pi*180),
                              (dop)*255, inten/inten.max()*255)))

    # hsv = np.uint8(cv2.merge(((aop+np.pi/2)/np.pi*180,
    #                          dop*255,
    #                          inten/inten.max()*255)))
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb

    # nbr, nbc = rgb.shape[0], rgb.shape[1]
    # fina = np.zeros((nbr*2, nbc*2, 3), dtype='uint8')
    # aop_colorHSV = np.uint8(cv2.merge(((aop+np.pi/2)/np.pi*180,
    #                                   np.ones(aop.shape)*255,
    #                                   np.ones(aop.shape)*255)))
    # aop_colorRGB = cv2.cvtColor(aop_colorHSV, cv2.COLOR_HSV2RGB)

    # for c in range(3):
    #    fina[:nbr, :nbc, c] = np.uint8(inten/inten.max()*255)
    #    fina[:nbr, nbc:, c] = aop_colorRGB[:, :, c]
    #    fina[nbr:, :nbc, c] = np.uint8(dop*255)
    #    fina[nbr:, nbc:, c] = self.rgb[:, :, c]
    # return fina

    # return rgb
