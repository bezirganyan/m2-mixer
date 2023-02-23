import pytest
import torch

from modules import ConcatFusion, SumFusion, MaxFusion, MeanFusion, BiModalGatedUnit, MultiModalGatedUnit, \
    ConcatDynaFusion


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

        assert fusion.get_output_shape(input_1.shape, input_2.shape) == (10, 40, 30)
        assert fusion.get_output_shape(20, 20, dim=1) == 40
        assert fusion.get_output_shape(20, 20, dim=0) == 20
        with pytest.raises(ValueError):
            fusion.get_output_shape(input_1, input_2, dim=2)

    def test_concat_dyna_fusion(self):
        fusion = ConcatDynaFusion(useless_arg=1)
        input_1 = torch.rand(10, 20, 20, 30)
        input_2 = torch.rand(10, 20, 20, 30)
        assert fusion(input_1, input_2).shape == (10, 40, 40, 30)

        assert fusion.get_output_shape(input_1.shape, input_2.shape) == (10, 40, 40, 30)
        assert fusion.get_output_shape(36, 36, dim=1) == 12*12
        assert fusion.get_output_shape(20, 20, dim=0) == 20
        with pytest.raises(ValueError):
            fusion.get_output_shape(input_1, input_2, dim=2)

    def test_sum_fusion(self):
        fusion = SumFusion(useless_arg=1)
        input_1 = torch.rand(10, 20, 30)
        input_2 = torch.rand(10, 20, 30)
        assert fusion(input_1, input_2).shape == (10, 20, 30)
        assert fusion.get_output_shape(input_1.shape, input_2.shape) == (10, 20, 30)
        assert fusion.get_output_shape(20, 20, dim=1) == 20
        assert fusion.get_output_shape(20, 20, dim=0) == 20
        with pytest.raises(ValueError):
            fusion.get_output_shape(input_1, input_2, dim=2)

    def test_max_fusion(self):
        fusion = MaxFusion(useless_arg=1)
        input_1 = torch.rand(10, 20, 30)
        input_2 = torch.rand(10, 20, 30)
        assert fusion.get_output_shape(input_1.shape, input_2.shape) == (10, 20, 30)
        assert fusion(input_1, input_2).shape == (10, 20, 30)
        assert fusion.get_output_shape(20, 20, dim=1) == 20
        assert fusion.get_output_shape(20, 20, dim=0) == 20
        with pytest.raises(ValueError):
            fusion.get_output_shape(input_1, input_2, dim=2)

    def test_mean_fusion(self):
        fusion = MeanFusion(useless_arg=1)
        input_1 = torch.rand(10, 20, 30)
        input_2 = torch.rand(10, 20, 30)
        assert fusion(input_1, input_2).shape == (10, 20, 30)
        assert fusion.get_output_shape(input_1.shape, input_2.shape) == (10, 20, 30)
        assert fusion.get_output_shape(20, 20, dim=1) == 20
        assert fusion.get_output_shape(20, 20, dim=0) == 20
        with pytest.raises(ValueError):
            fusion.get_output_shape(input_1, input_2, dim=2)

    def test_bi_modal_gu_fusion(self):
        input_1 = torch.rand(10, 20, 30)
        input_2 = torch.rand(10, 20, 30)
        fusion = BiModalGatedUnit(30, 30, 30, useless_arg=1)
        assert fusion(input_1, input_2).shape == (10, 20, 30)
        assert fusion.get_output_shape(input_1.shape, input_2.shape) == (10, 20, 30)
        assert fusion.get_output_shape(20, 20, dim=1) == 20
        assert fusion.get_output_shape(20, 20, dim=-1) == 30
        with pytest.raises(ValueError):
            fusion.get_output_shape(input_1, input_2, dim=2)

    def test_multimodal_gu_fusion(self):
        input_1 = torch.rand(10, 20, 30)
        input_2 = torch.rand(10, 20, 30)
        input_3 = torch.rand(10, 20, 30)
        fusion = MultiModalGatedUnit(30, 30, 30, out_size=30, useless_arg=1)
        assert fusion(input_1, input_2, input_3).shape == (10, 20, 30)
        assert fusion.get_output_shape(input_1.shape, input_2.shape) == (10, 20, 30)
        assert fusion.get_output_shape(20, 20, 20, dim=1) == 20
        assert fusion.get_output_shape(20, 20, 20, dim=-1) == 30
        with pytest.raises(ValueError):
            fusion.get_output_shape(input_1, input_2, input_3, dim=2)
