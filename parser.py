import re
from abc import ABCMeta, abstractmethod

import numpy as np


class Parser(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def get_parser_name():
        pass

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @classmethod
    def get_parser(cls, **kwargs):
        return cls(**kwargs)

    @abstractmethod
    def parse(self, code) -> None:
        pass

    @abstractmethod
    def get_array(self) -> np.ndarray:
        pass

    @staticmethod
    def cpp(code):
        return re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code).strip()

    @staticmethod
    @abstractmethod
    def similarity(arr1, arr2) -> float:
        pass
