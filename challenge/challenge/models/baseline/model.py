import torch
import torch.nn as nn

from challenge.base import ModelBase
from challenge.utils import setup_logger


log = setup_logger(__name__)


class Baseline(ModelBase):
    def __init__(self, in_features: int):
        """ Simple baseline model for prediction secondary structure
        Args:
            in_features: size in features
        """
        super(Baseline, self).__init__()

        # Task block
        self.ss8_fc1 = nn.Linear(in_features=in_features, out_features=512)
        self.ss8_fc2 = nn.Linear(in_features=512, out_features=512)
        self.ss8_out = nn.Linear(in_features=512, out_features=8)

        self.ss3_fc1 = nn.Linear(in_features=in_features, out_features=512)
        self.ss3_fc2 = nn.Linear(in_features=512, out_features=512)
        self.ss3_out = nn.Linear(in_features=512, out_features=3)

        log.info(f'<init>: \n{self}')

    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """ Forwarding logic """

        ss8 = self.ss8_fc1(x)
        ss8 = self.ss8_fc2(ss8)
        ss8 = self.ss8_out(ss8)

        ss3 = self.ss3_fc1(x)
        ss3 = self.ss3_fc2(ss3)
        ss3 = self.ss3_out(ss3)


        return [ss8, ss3]
