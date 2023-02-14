import pytest
import torch

from modules import ConcatFusion, SumFusion, MaxFusion, MeanFusion, BiModalGatedUnit


class TestFusions:
    """
    Fusions must be able to accept at least 2 inputs, and provide the fused output.
    At initialization, the fusion must be able to accept additional useless arguments.
    """

    def test_concat_fusion(self):
        fusion = ConcatFusion(useless_arg=1)
        input_1 = torch.rand(10, 20, 30)
        input_2 = torch.rand(10, 20, 30)
        assert fusion(input_1, input_2).shape == (10, 40, 30)

    def test_sum_fusion(self):
        fusion = SumFusion(useless_arg=1)
        input_1 = torch.rand(10, 20, 30)
        input_2 = torch.rand(10, 20, 30)
        assert fusion(input_1, input_2).shape == (10, 20, 30)

    def test_max_fusion(self):
        fusion = MaxFusion(useless_arg=1)
        input_1 = torch.rand(10, 20, 30)
        input_2 = torch.rand(10, 20, 30)
        assert fusion(input_1, input_2).shape == (10, 20, 30)

    def test_mean_fusion(self):
        fusion = MeanFusion(useless_arg=1)
        input_1 = torch.rand(10, 20, 30)
        input_2 = torch.rand(10, 20, 30)
        assert fusion(input_1, input_2).shape == (10, 20, 30)

    def test_bi_modal_gu_fusion(self):
        input_1 = torch.rand(10, 20, 30)
        input_2 = torch.rand(10, 20, 30)
        fusion = BiModalGatedUnit(30, 30, 30, useless_arg=1)
        assert fusion(input_1, input_2).shape == (10, 20, 30)
