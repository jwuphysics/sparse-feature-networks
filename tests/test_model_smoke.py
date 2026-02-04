import torch

from src.model import ResNet18TopK


def test_resnet18_topk_forward_shapes():
    torch.manual_seed(0)

    model = ResNet18TopK(k=4, n_out=3, pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        y = model(x)
        a = model.sparse_features(x)

    assert y.shape == (2, 3)
    assert a.shape[0] == 2
    assert a.ndim == 2


def test_resnet18_topk_sparsity_upper_bound():
    torch.manual_seed(0)

    k = 5
    model = ResNet18TopK(k=k, n_out=1, pretrained=False)
    model.eval()

    x = torch.randn(8, 3, 224, 224)

    with torch.no_grad():
        a = model.sparse_features(x)

    # After top-k, each row should have at most k non-zero entries.
    # (If some top-k values are exactly 0 after ReLU, nnz can be < k.)
    nnz = (a != 0).sum(dim=1)
    assert int(nnz.max()) <= k
