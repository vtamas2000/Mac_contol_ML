import cv2
import os, re, time, argparse
import torch
import torchvision
from typing import List


def get_agrs() -> argparse.Namespace:
    """
    Collects and parses command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--motion",
        type=str,
        required=True,
        choices=["up", "down", "left", "right", "default"],
        help="Which motion to capture, select from up, down, left, right",
    )

    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="data",
        help="Directory where the images will be saved",
    )

    parser.add_argument(
        "-s", "--size", type=int, default=800, help="Size of the center crop"
    )

    parser.add_argument(
        "--name_base",
        type=str,
        default="picture",
        help="The base of the name to use for naming saved pictures",
    )

    args = parser.parse_args()
    return args


def find_available_file_name(path: str, name_base: str) -> str:
    """
    Finds the next available filename in `path` and retuns it.
    Assumes that all files in the directory have the same file extension and name base

    :param path: Directory to look at
    :param name_base: The base of the name of the files
    (e.g.: if there are example1.png, example2.png then the `name_base` is example)
    :return: Name of the next available file name
    """

    files = os.listdir(path)
    if len(files) == 0:
        return name_base + "0.png"
    else:
        file_numbers = []
        for file in files:
            file_number = int(re.match(f"^{name_base}(\d+).png", file).group(1))
            file_numbers.append(file_number)

        return name_base + str(max(file_numbers) + 1) + ".png"


def collect_image(camera: cv2.VideoCapture, size: int = 400) -> torch.Tensor | None:
    """
    Collects and returns an image from the camera for training
    :param camera: cv2 Videocapture object
    :param size: Size to be cropped from center of image
    :returns: Tensor of image data
    """

    result, image = camera.read()

    if result:
        img_tensor = torch.tensor(image).permute(2, 0, 1)
        cropped_img = torchvision.transforms.CenterCrop(size)(img_tensor)
        numpy_image = cropped_img.permute(1, 2, 0).numpy()
        cv2.imshow("Image", numpy_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return torchvision.transforms.functional.to_tensor(numpy_image)
    else:
        print("No image was captured")
        return None


def main() -> None:
    args = get_agrs()

    target_dir = os.path.join(args.directory, args.motion)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    camera = cv2.VideoCapture(0)
    time.sleep(1)

    while True:
        image = collect_image(camera, args.size)
        image_name = find_available_file_name(target_dir, args.name_base)
        if image is not None:
            torchvision.utils.save_image(image, os.path.join(target_dir, image_name))
            print(f"{image_name} was succesfully saved")
        else:
            print(f"{image_name} was not captured")


if __name__ == "__main__":
    main()
