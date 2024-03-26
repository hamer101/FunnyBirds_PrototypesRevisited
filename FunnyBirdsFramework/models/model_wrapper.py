import torch.nn as nn
from abc import abstractmethod

class ModelExplainerWrapper:

    def __init__(self, model, explainer):
        """
        A generic wrapper that takes any model and any explainer to putput model predictions 
        and explanations that highlight important input image part.
        Args:
            model: PyTorch neural network model
            explainer: PyTorch model explainer    
        """
        self.model = model
        self.explainer = explainer

    def predict(self, input):
        return self.model.forward(input)

    def explain(self, input):
        return self.explainer.explain(self.model, input)


class AbstractModel(nn.Module):
    def __init__(self, model):
        """
        An abstract wrapper for PyTorch models implementing functions required for evaluation.
        Args:
            model: PyTorch neural network model
        """
        super().__init__()
        self.model = model

    @abstractmethod
    def forward(self, input):
        return self.model

class StandardModel(AbstractModel):
    """
    A wrapper for standard PyTorch models (e.g. ResNet, VGG, AlexNet, ...).
    Args:
        model: PyTorch neural network model
    """

    def forward(self, input):
        return self.model(input)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

class ProtoPNetWrapper(AbstractModel):
    """
    A wrapper for ProtoPNet models.
    Args:
        model: ProtoPNet model
    """

    # Overriding the output of model since it returns a tuple (logis and prototypes) of sizes
    # torch.Size([8, 50]) torch.Size([8, 500])
    def forward(self, input):
        return self.model(input)[0]