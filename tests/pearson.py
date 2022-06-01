from modeling import build_model
from net_config import cfg
import torch
import csv
from scipy.stats import pearsonr

TYPE_VALUE_MAP = {
    'EQUI': 5,
    'OPPO': 4,
    'SPE1': 3,
    'SPE2': 3,
    'SIMI': 2,
    'REL': 1
}

class PearsonMetrics:
    def __init__(self, weights_file, test_file):
        self.weights_file = weights_file
        self.test_file = test_file
        model = build_model(cfg)
        model.load_state_dict(torch.load(self.weights_file))
        self.model = model

    def predict(self):
        self._predicted_types = []
        self._predicted_scores = []
        self._target_types = []
        self._target_scores = []
        with open(self.test_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                if i > 0 and line:
                    try:
                        parts = line[0].split(",")
                        tokens = self.model.roberta.encode(parts[0], parts[1])
                        _, _, out3 = self.model(tokens)
                        _, result = torch.max(out3, 1)
                        score, type = self.decode_class(result.item())
                        self._target_scores.append(int(parts[2]))
                        self._target_types.append(TYPE_VALUE_MAP[parts[3]])
                        self._predicted_scores.append(score)
                        self._predicted_types.append(TYPE_VALUE_MAP[type])
                    except Exception as e:
                        print(e)

    def print_statistics(self):
        print("[Pearson Score], correlation for alignment type labels: {}".format(pearsonr(self._target_types, self._predicted_types)))
        print("[Pearson Type], correlation for alignment score labels: {}".format(pearsonr(self._target_scores, self._predicted_scores)))

    def decode_class(self, value_type):
        if value_type in [0, 1, 2, 3]:
            value = 1
        elif value_type in [4, 5, 6, 7]:
            value = 2
        elif value_type in [8, 9, 10, 11]:
            value = 3
        elif value_type in [12, 13, 14, 15, 16, 17]:
            value = 4
        else:
            value = 5

        if value_type in [0, 4, 8, 14]:
            exp = "REL"
        elif value_type in [1, 5, 9, 15]:
            exp = "SIMI"
        elif value_type in [2, 6, 10, 16]:
            exp = "SPE1"
        elif value_type in [3, 7, 11, 17]:
            exp = "SPE2"
        elif value_type in [12, 18]:
            exp = "EQUI"
        else:
            exp = "OPPO"

        return value, exp

if __name__ == "__main__":
    pearson = PearsonMetrics("output/150_model.pt", "data/datasets/test_answers-students.csv")
    pearson.predict()
    pearson.print_statistics()