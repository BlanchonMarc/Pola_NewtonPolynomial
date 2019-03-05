import argparse
import os
from PIL import Image
import numpy as np
from process import process_interpolation
from process import fill_values
from process import interpolate_pol
from process import convert_to_HSL
import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input path",
                        type=str)
    parser.add_argument("-o", "--output", help="output path")
    args = parser.parse_args()
    print(f"input path is: {args.input}")
    print(f"ouput path is: {args.output}")
    input_images_names = os.listdir(args.input)

    input_images_path = [f"{args.input}{x}" for x in input_images_names]

    output_images_path = [f"{args.output}{x}" for x in input_images_names]
    # print(input_images_path)
    # print(output_images_path)

    for indx, path in enumerate(input_images_path):
        # image = np.array(Image.open(path))
        # print(image.shape)
        image = cv2.imread(path, -1)
        if image is not None:
            i0, i45, i135, i90 = process_interpolation(image)

            i0, i45, i135, i90 = fill_values(i0, i45, i135, i90, image)

            i0, i45, i135, i90 = interpolate_pol(i0, i45, i135, i90)

            HSL = convert_to_HSL(i0, i45, i135, i90)

            cv2.imwrite(output_images_path[indx], HSL)
            print('Done')
        # image_out = Image.fromarray(HSL)
        # image_out.save(output_images_path[indx])
        # image_out = Image.fromarray(i45)
        # image_out.save("output/45.jpg")
        # image_out = Image.fromarray(i90)
        # image_out.save("output/90.jpg")
        # image_out = Image.fromarray(i135)
        # image_out.save("output/135.jpg")
