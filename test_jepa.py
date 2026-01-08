import unittest
import torch
from jepa_core import SIGRegLoss, VectorFlow

class TestJepaCore(unittest.TestCase):
    def setUp(self):
        self.input_dim = 64
        self.batch_size = 32
        self.sig_loss = SIGRegLoss(self.input_dim)
        self.vector_flow = VectorFlow(self.input_dim)

    def test_sigreg_gaussian(self):
        x = torch.randn(self.batch_size, self.input_dim)
        loss = self.sig_loss(x)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)

    def test_sigreg_uniform(self):
        x = torch.rand(self.batch_size, self.input_dim)
        loss = self.sig_loss(x)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)

    def test_vector_flow_shapes(self):
        x = torch.randn(self.batch_size, self.input_dim)
        output = self.vector_flow(x)
        self.assertEqual(output.shape, (self.batch_size, self.input_dim))

if __name__ == '__main__':
    unittest.main()