import os
from model.MSTNet import MSTNet

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "synthetic_dataset"))

used_model = MSTNet

