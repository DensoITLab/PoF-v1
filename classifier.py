import torch.nn as nn
import torch.nn.init as init

class Classifier(nn.Module):
    def __init__(self, numFeatureDim=None, numClass=None, PretrainedParam=None):
        super(Classifier, self).__init__()
        if PretrainedParam is not None:
            self.classifiers = PretrainedParam
        else:  
            self.classifiers = self._make_layers(numFeatureDim, numClass)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight.data, mean=0, std=0.1)
                    m.bias.data.zero_()

    def _make_layers(self, numFeatureDim, numClass):
        layers = []
        layers += [nn.Linear(numFeatureDim, numClass)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.classifiers(x)
