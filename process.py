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
                            I45 = image[indx, indy]
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

                            I_45 = image[indx, indy]
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

                d45 = abs(image[xvec[2], yvec[1]] - image[xvec[1], yvec[2]]) + abs(
                    2*image[x, y] - image[xvec[0], yvec[3]] - image[xvec[3], yvec[0]])
                d_45 = abs(image[xvec[1], yvec[1]]-image[xvec[2], yvec[2]]) + abs(
                    2*image[x, y] - image[xvec[0], yvec[0]] - image[xvec[3], yvec[3]])

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

    inter1 = Image.fromarray(i0)
    img = inter1.resize((int(640 / 2), int(480 / 2)), Image.ANTIALIAS)
    inter2 = img
    img = inter2.resize((int(640), int(480)), Image.ANTIALIAS)

    i0 = np.array(img)

    inter1 = Image.fromarray(i45)
    img = inter1.resize((int(640 / 2), int(480 / 2)), Image.ANTIALIAS)
    inter2 = img
    img = inter2.resize((int(640), int(480)), Image.ANTIALIAS)

    i45 = np.array(img)

    inter1 = Image.fromarray(i135)
    img = inter1.resize((int(640 / 2), int(480 / 2)), Image.ANTIALIAS)
    inter2 = img
    img = inter2.resize((int(640), int(480)), Image.ANTIALIAS)

    i135 = np.array(img)

    inter1 = Image.fromarray(i90)
    img = inter1.resize((int(640 / 2), int(480 / 2)), Image.ANTIALIAS)
    inter2 = img
    img = inter2.resize((int(640), int(480)), Image.ANTIALIAS)

    i90 = np.array(img)

    return [i0, i45, i135, i90]
