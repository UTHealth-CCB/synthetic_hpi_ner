import os
import re
from utils.metrics.Metrics import Metrics


class NumA(Metrics):
    def __init__(self, test_text, token):
        super().__init__()
        self.name = 'Num_'+ token
        self.test_data = test_text
        self.pattern = r"^" + token + r" "

    def get_score(self):
        num = 0
        with open(self.test_data) as fin:
            c = fin.readlines()
        for line in c:
            res = re.findall(self.pattern, line)
            num += len(res)
        return num