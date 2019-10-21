import unittest
import torch

from modules.faster_rcnn import MaskRCNN_Benchmark


class MyTestCase(unittest.TestCase):
    def test_faster_rcnn(self):
        faster_rcnn = MaskRCNN_Benchmark()
        test_input = [torch.rand(3, 255, 255), torch.rand(3, 255, 255)]
        features = faster_rcnn(test_input)
        self.assertEqual(features.shape, torch.Size([2, 100, 2048]))


if __name__ == '__main__':
    unittest.main()
