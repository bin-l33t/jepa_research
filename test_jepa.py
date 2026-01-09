# AGENT: VALIDATOR
import unittest
from jepa_core import SIGRegLoss, VectorFlow

class TestJEPA(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.dim = 64
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_sigreg_loss_gaussian(self):
        """Test SIGRegLoss on Gaussian input (torch.randn)."""
        x = torch.randn(self.batch_size, self.dim, device=self.device)
        y = torch.randn(self.batch_size, self.dim, device=self.device)
        
        criterion = SIGRegLoss()
        loss = criterion(x, y)
        
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.ndim, 0)

    def test_sigreg_loss_uniform(self):
        """Test SIGRegLoss on Uniform input (torch.rand)."""
        x = torch.rand(self.batch_size, self.dim, device=self.device)
        y = torch.rand(self.batch_size, self.dim, device=self.device)
        
        criterion = SIGRegLoss()
        loss = criterion(x, y)
        
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.ndim, 0)

    def test_vector_flow_shapes(self):
        """Check VectorFlow output shapes match input shapes."""
        # Try instantiating with dim arg, fallback to no arg if necessary
        try:
            model = VectorFlow(self.dim).to(self.device)
        except TypeError:
            model = VectorFlow().to(self.device)
            
        x = torch.randn(self.batch_size, self.dim, device=self.device)
        output = model(x)
        
        self.assertEqual(output.shape, x.shape)

if __name__ == '__main__':
    unittest.main()