import unittest
import os
import sys
import io
from unittest.mock import patch, MagicMock
import json
import numpy as np

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from main.py to test
from main import (
    get_image_metadata,
    analyze_dct_coefficients,
    analyze_noise_patterns,
    analyze_frequency_domain,
    combine_analysis_results,
    make_serializable
)

class TestImageAnalysis(unittest.TestCase):
    """Test individual analysis functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a small test image in memory
        self.test_image_data = self._create_test_jpeg()
        
        # Mock metadata for testing combine_analysis
        self.mock_metadata = {
            "format": "JPEG",
            "mode": "RGB",
            "size": [100, 100]
        }
        
        self.mock_dct_analysis = {
            "dc_mean": 1000.0,
            "dc_std": 500.0,
            "ac_mean": 0.2,
            "ac_std": 8.0,
            "histogram": [0.1, 0.2, 0.3, 0.4],
            "block_count": 100
        }
        
        self.mock_noise_analysis = {
            "blue_channel": {"mean": 1.5, "std": 2.5},
            "green_channel": {"mean": 1.4, "std": 2.6},
            "red_channel": {"mean": 1.3, "std": 2.4},
            "laplacian": {"mean": 4.0, "std": 4.5},
            "noise_level": 2.5
        }
        
        self.mock_frequency_analysis = {
            "energy_low_freq": 0.3,
            "energy_mid_freq": 0.5,
            "energy_high_freq": 0.2,
            "energy_ratio_high_low": 0.67
        }
    
    def _create_test_jpeg(self):
        """Create a small test JPEG image"""
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='white')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        return img_byte_arr.getvalue()
    
    def test_get_image_metadata(self):
        """Test metadata extraction function"""
        metadata = get_image_metadata(self.test_image_data)
        
        # Check basic metadata fields
        self.assertEqual(metadata["format"], "JPEG")
        self.assertEqual(metadata["mode"], "RGB")
        self.assertEqual(len(metadata["size"]), 2)
        self.assertEqual(metadata["size"][0], 100)
        self.assertEqual(metadata["size"][1], 100)
    
    def test_analyze_dct_coefficients(self):
        """Test DCT coefficient analysis"""
        dct_analysis = analyze_dct_coefficients(self.test_image_data)
        
        # Check that required keys exist and have appropriate types
        self.assertIn("dc_mean", dct_analysis)
        self.assertIn("dc_std", dct_analysis)
        self.assertIn("ac_mean", dct_analysis)
        self.assertIn("ac_std", dct_analysis)
        self.assertIn("histogram", dct_analysis)
        self.assertIn("block_count", dct_analysis)
        
        self.assertIsInstance(dct_analysis["dc_mean"], float)
        self.assertIsInstance(dct_analysis["histogram"], list)
        self.assertIsInstance(dct_analysis["block_count"], int)
    
    def test_analyze_noise_patterns(self):
        """Test noise pattern analysis"""
        noise_analysis = analyze_noise_patterns(self.test_image_data)
        
        # Check that required keys exist and have appropriate types
        self.assertIn("blue_channel", noise_analysis)
        self.assertIn("green_channel", noise_analysis)
        self.assertIn("red_channel", noise_analysis)
        self.assertIn("laplacian", noise_analysis)
        self.assertIn("noise_level", noise_analysis)
        
        self.assertIsInstance(noise_analysis["noise_level"], float)
        self.assertIn("mean", noise_analysis["blue_channel"])
        self.assertIn("std", noise_analysis["blue_channel"])
    
    def test_analyze_frequency_domain(self):
        """Test frequency domain analysis"""
        frequency_analysis = analyze_frequency_domain(self.test_image_data)
        
        # Check that required keys exist and have appropriate types
        self.assertIn("energy_low_freq", frequency_analysis)
        self.assertIn("energy_mid_freq", frequency_analysis)
        self.assertIn("energy_high_freq", frequency_analysis)
        self.assertIn("energy_ratio_high_low", frequency_analysis)
        
        self.assertIsInstance(frequency_analysis["energy_low_freq"], float)
        
        # Energy distribution should sum to approximately 1
        total_energy = (frequency_analysis["energy_low_freq"] + 
                        frequency_analysis["energy_mid_freq"] + 
                        frequency_analysis["energy_high_freq"])
        self.assertAlmostEqual(total_energy, 1.0, delta=0.01)
    
    def test_combine_analysis_results_ai_positive(self):
        """Test detection logic with parameters that should trigger AI detection"""
        # Modify mock data to have AI characteristics
        ai_noise_analysis = self.mock_noise_analysis.copy()
        ai_noise_analysis["noise_level"] = 1.5  # Low noise (AI characteristic)
        ai_noise_analysis["laplacian"]["std"] = 3.0  # Low edge detail
        
        ai_dct_analysis = self.mock_dct_analysis.copy()
        ai_dct_analysis["dc_mean"] = 10000.0
        ai_dct_analysis["ac_mean"] = 0.001  # Very high DC/AC ratio
        
        result = combine_analysis_results(
            self.mock_metadata,
            ai_dct_analysis,
            ai_noise_analysis,
            self.mock_frequency_analysis
        )
        
        # Should detect as AI
        self.assertTrue(result["is_ai_generated"])
        self.assertGreater(result["confidence"], 0.6)
        self.assertGreater(len(result["indicators"]), 1)
    
    def test_combine_analysis_results_ai_negative(self):
        """Test detection logic with parameters that should indicate a real image"""
        # Modify mock data to have real image characteristics
        real_noise_analysis = self.mock_noise_analysis.copy()
        real_noise_analysis["noise_level"] = 8.0  # High noise (real image characteristic)
        real_noise_analysis["laplacian"]["std"] = 12.0  # High edge detail
        
        real_dct_analysis = self.mock_dct_analysis.copy()
        real_dct_analysis["ac_std"] = 20.0  # Higher AC variation (real image)
        
        real_freq_analysis = self.mock_frequency_analysis.copy()
        real_freq_analysis["energy_ratio_high_low"] = 0.8  # More balanced frequency distribution
        
        result = combine_analysis_results(
            self.mock_metadata,
            real_dct_analysis,
            real_noise_analysis,
            real_freq_analysis
        )
        
        # Should detect as real (not AI)
        self.assertFalse(result["is_ai_generated"])
        self.assertLess(result["confidence"], 0.6)
    
    def test_combine_analysis_results_heic_format(self):
        """Test that HEIC format is treated as more likely to be real"""
        heic_metadata = self.mock_metadata.copy()
        heic_metadata["format"] = "HEIC"
        
        # Use parameters that would typically indicate AI
        result = combine_analysis_results(
            heic_metadata,
            self.mock_dct_analysis,
            self.mock_noise_analysis,
            self.mock_frequency_analysis
        )
        
        # There should be an indicator about HEIC format
        self.assertTrue(any("HEIC" in indicator for indicator in result["indicators"]))
        
        # Compare with identical parameters but JPEG format
        jpeg_result = combine_analysis_results(
            self.mock_metadata,  # JPEG format
            self.mock_dct_analysis,
            self.mock_noise_analysis,
            self.mock_frequency_analysis
        )
        
        # HEIC should have lower confidence of being AI
        self.assertLess(result["confidence"], jpeg_result["confidence"])
    
    def test_make_serializable(self):
        """Test serialization helper function"""
        # Test various types
        test_data = {
            "string": "test",
            "int": 123,
            "float": 123.45,
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": 2},
            "none": None,
            "bytes": b"test bytes",
            "complex": complex(1, 2)
        }
        
        # Mock a rational number (like in EXIF)
        class MockRational:
            def __init__(self, num, denom):
                self.numerator = num
                self.denominator = denom
        
        test_data["rational"] = MockRational(1, 2)
        test_data["rational_zero_denom"] = MockRational(1, 0)
        
        result = make_serializable(test_data)
        
        # Check serialized values
        self.assertEqual(result["string"], "test")
        self.assertEqual(result["int"], 123)
        self.assertEqual(result["float"], 123.45)
        self.assertEqual(result["list"], [1, 2, 3])
        self.assertEqual(result["nested"], {"a": 1, "b": 2})
        self.assertIsNone(result["none"])
        self.assertEqual(result["bytes"], "test bytes")
        self.assertEqual(result["rational"], 0.5)
        self.assertEqual(type(result["complex"]), str)
        
        # Test JSON serialization
        try:
            json.dumps(result)
            serializable = True
        except:
            serializable = False
        self.assertTrue(serializable, "Result should be JSON serializable")


class TestErrorHandling(unittest.TestCase):
    """Test error handling in various functions"""
    
    def test_metadata_error_handling(self):
        """Test handling of invalid image data in metadata function"""
        invalid_data = b"not an image"
        result = get_image_metadata(invalid_data)
        self.assertIn("error", result)
    
    def test_dct_error_handling(self):
        """Test handling of invalid image data in DCT analysis"""
        invalid_data = b"not an image"
        result = analyze_dct_coefficients(invalid_data)
        self.assertIn("error", result)
    
    def test_noise_error_handling(self):
        """Test handling of invalid image data in noise analysis"""
        invalid_data = b"not an image"
        result = analyze_noise_patterns(invalid_data)
        self.assertIn("error", result)
    
    def test_frequency_error_handling(self):
        """Test handling of invalid image data in frequency analysis"""
        invalid_data = b"not an image"
        result = analyze_frequency_domain(invalid_data)
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main() 