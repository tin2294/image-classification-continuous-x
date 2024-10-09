import unittest
import os
import sys
import numpy as np
import tempfile  # Import tempfile to create temporary directories
from PIL import Image  # Import Image from PIL
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_training_labels

class TestUtils(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory structure for testing."""
        self.test_directory = tempfile.mkdtemp()
        self.valid_class_dir = os.path.join(self.test_directory, 'class_0')
        os.makedirs(self.valid_class_dir)

        img = Image.new('RGB', (100, 100), color='red')
        img.save(os.path.join(self.valid_class_dir, 'sample_image.jpg'))

    def tearDown(self):
        """Clean up the temporary directory after tests."""
        shutil.rmtree(self.test_directory)

    def test_load_training_labels(self):
        test_directory = os.path.join(os.path.dirname(__file__), '..', 'test_data')
        images, labels = load_training_labels(test_directory)

        self.assertIsInstance(images, list)
        self.assertIsInstance(labels, np.ndarray)

if __name__ == '__main__':
    unittest.main()
