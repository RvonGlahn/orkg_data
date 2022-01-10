import logging
import os
import sys


class MyLogger:

    def __init__(self, filename: str):
        dir_path = os.path.dirname(__file__)
        path = os.path.join(dir_path, filename + '.log')

        # set logger and global logging threshold
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.INFO)

        # StreamHandler for shell
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        # FileHandler for File
        filehandler = logging.FileHandler(path)
        filehandler.setLevel(logging.INFO)

        # Create Formatter
        stream_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # set formatter in handler
        stream_handler.setFormatter(stream_format)
        filehandler.setFormatter(file_format)

        # Add Handler to logger
        self.logger.addHandler(stream_handler)
        self.logger.addHandler(filehandler)



