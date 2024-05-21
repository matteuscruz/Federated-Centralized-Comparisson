import torch
import warnings
import flwr as fl
from fastai.vision.all import *
from collections import OrderedDict

warnings.filterwarnings("ignore", category=UserWarning)

# Define the dataset
path = 'dataset/pneumonia-adult/set_2'

epochs = 10

# Load dataset
dls = ImageDataLoaders.from_folder(
    path, valid_pct=0.5, train="training", valid="testing", num_workers=0
)

""""
p_path=Path(path)
fns = get_image_files(path)

class DataLoaders(GetAttr):
    def __init__(self, *loaders): self.loaders = loaders
    def __getitem__(self, i): return self.loaders[i]
    train,valid = add_props(lambda i,self: self[i])

bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))

failed = verify_images(fns)

dls = bears.dataloaders(path)
"""

# Define model
learn = vision_learner(dls, squeezenet1_1, metrics=[error_rate, accuracy], opt_func=Adam, lr=0.001, cut=None)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in learn.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(learn.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        learn.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        learn.fit(epochs)
        return self.get_parameters(config={}), len(dls.train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, error_rate = learn.validate()
        return loss, len(dls.valid), {"accuracy": 1 - error_rate}


# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient().to_client(),
)
