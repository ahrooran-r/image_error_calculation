import cv2
import glob
import pathlib
import pandas as pd
import torch
import piq


class ImageAnalysis:

    def __init__(self, source_path="./source/", target_path="./target/", ext=["jpg", "png"]):
        self.__source, self.__target = ImageAnalysis.__setup(source_path, target_path, ext)
        self.df = pd.DataFrame(columns=["File_Name", "PSNR", "SSIM"])

    def build(self, kernel_size=3, reduction="none", data_range=1.0):
        for i in range(len(self.__source)):
            src_image = torch.tensor(cv2.imread(self.__source[i]))[None, ...] / 255
            target_image = torch.tensor(cv2.imread(self.__target[i]))[None, ...] / 255

            file_name = pathlib.Path(self.__source[i]).stem
            psnr = piq.psnr(src_image, target_image, data_range=data_range, reduction=reduction).item()
            ssim = piq.ssim(src_image, target_image, data_range=data_range, kernel_size=kernel_size,
                            reduction=reduction).item()

            row = pd.Series([file_name, psnr, ssim], index=self.df.columns)
            self.df = self.df.append(row, ignore_index=True)

    def summary(self):
        avg_psnr = self.df["PSNR"].mean()
        avg_ssim = self.df["SSIM"].mean()
        return avg_psnr, avg_ssim

    @staticmethod
    def __setup(source_path, target_path, ext):
        """https://stackoverflow.com/a/52944780"""

        source_files = []
        [source_files.extend(glob.glob(source_path + '*.' + e)) for e in ext]

        target_files = []
        [target_files.extend(glob.glob(target_path + '*.' + e)) for e in ext]

        ImageAnalysis.__validate(source_files, target_files)

        return source_files, target_files

    @staticmethod
    def __validate(source_files, target_files):
        result = len(source_files) == len(target_files)
        if not result:
            raise AssertionError("Amount of files in 'source dir' and 'target dir' are not equal")
        else:
            for i in range(len(source_files)):
                ImageAnalysis.__validate_filename(source_files[i], target_files[i])

    @staticmethod
    def __validate_filename(source_path, target_path):
        """https://stackoverflow.com/a/47496703"""

        result = pathlib.Path(source_path).stem == pathlib.Path(target_path).stem
        if not result:
            raise AssertionError("File names do not match.\nMake sure all files in both 'source dir' and 'target "
                                 "dir' are in same order and their file names match with each other")