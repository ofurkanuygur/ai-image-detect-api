import unittest
import os
import requests
import io
import time
from PIL import Image
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("integration_tests")

# API base URL
API_BASE_URL = "http://localhost:8000/api"

def create_test_image(width=100, height=100, color=(255, 255, 255), format="JPEG"):
    """Create a test image in memory"""
    img = Image.new('RGB', (width, height), color=color)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format)
    img_byte_arr.seek(0)
    return img_byte_arr

class APIIntegrationTests(unittest.TestCase):
    """Test cases for API endpoints"""

    def setUp(self):
        """Check if API is running before tests"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code != 200:
                self.skipTest("API not running or not healthy")
        except requests.exceptions.RequestException:
            self.skipTest("API not running")
    
    def test_health_endpoint(self):
        """Test that health endpoint is working"""
        response = requests.get(f"{API_BASE_URL}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("version", data)
        self.assertIn("timestamp", data)
    
    def test_detect_endpoint_with_jpeg(self):
        """Test detect endpoint with a JPEG image"""
        img_data = create_test_image(format="JPEG")
        files = {'image_file': ('test.jpg', img_data, 'image/jpeg')}
        response = requests.post(f"{API_BASE_URL}/detect", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check that all expected fields are in the response
        self.assertIn("is_ai_generated", data)
        self.assertIn("confidence", data)
        self.assertIn("indicators", data)
        self.assertIn("metadata", data)
        self.assertIn("dct_analysis", data)
        self.assertIn("noise_analysis", data)
        self.assertIn("frequency_analysis", data)
        
        # Metadata should identify the format correctly
        self.assertEqual(data["metadata"]["format"], "JPEG")
    
    def test_detect_endpoint_with_png(self):
        """Test detect endpoint with a PNG image"""
        img_data = create_test_image(format="PNG")
        files = {'image_file': ('test.png', img_data, 'image/png')}
        response = requests.post(f"{API_BASE_URL}/detect", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Metadata should identify the format correctly
        self.assertEqual(data["metadata"]["format"], "PNG")
    
    def test_detect_with_empty_file(self):
        """Test detect endpoint with an empty file (should return 400)"""
        files = {'image_file': ('empty.jpg', io.BytesIO(), 'image/jpeg')}
        response = requests.post(f"{API_BASE_URL}/detect", files=files)
        
        self.assertEqual(response.status_code, 400)
    
    def test_detect_with_non_image(self):
        """Test detect endpoint with a non-image file (should return 400)"""
        text_data = io.BytesIO(b"This is not an image")
        files = {'image_file': ('text.txt', text_data, 'text/plain')}
        response = requests.post(f"{API_BASE_URL}/detect", files=files)
        
        self.assertEqual(response.status_code, 400)
    
    def test_analysis_details_endpoint(self):
        """Test analysis_details endpoint"""
        img_data = create_test_image()
        files = {'image_file': ('test.jpg', img_data, 'image/jpeg')}
        response = requests.post(f"{API_BASE_URL}/analysis_details", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check that all expected fields are in the response
        self.assertIn("metadata", data)
        self.assertIn("dct_analysis", data)
        self.assertIn("noise_analysis", data)
        self.assertIn("frequency_analysis", data)
    
    def test_batch_endpoint(self):
        """Test batch processing endpoint"""
        # Create multiple test images
        img1 = create_test_image(width=100, height=100, color=(255, 255, 255))
        img2 = create_test_image(width=200, height=200, color=(0, 0, 0))
        
        files = [
            ('image_files', ('test1.jpg', img1.getvalue(), 'image/jpeg')),
            ('image_files', ('test2.jpg', img2.getvalue(), 'image/jpeg'))
        ]
        
        response = requests.post(f"{API_BASE_URL}/detect/batch", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Should return results for both images
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 2)
        
        # Each result should have the same structure as a single result
        for result in data["results"]:
            self.assertIn("is_ai_generated", result)
            self.assertIn("confidence", result)
            self.assertIn("indicators", result)
    
    def test_compare_endpoint(self):
        """Test compare endpoint"""
        # Create two different test images
        img1 = create_test_image(width=100, height=100, color=(255, 255, 255))
        img2 = create_test_image(width=200, height=200, color=(0, 0, 0))
        
        files = {
            'image_file1': ('test1.jpg', img1.getvalue(), 'image/jpeg'),
            'image_file2': ('test2.jpg', img2.getvalue(), 'image/jpeg')
        }
        
        response = requests.post(f"{API_BASE_URL}/compare", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check that all expected fields are in the response
        self.assertIn("image1", data)
        self.assertIn("image2", data)
        self.assertIn("summary", data)
        
        # Both images should have complete analysis results
        for img_key in ["image1", "image2"]:
            self.assertIn("is_ai_generated", data[img_key])
            self.assertIn("confidence", data[img_key])
            self.assertIn("indicators", data[img_key])
    
    def test_api_under_load(self):
        """Test API performance under load with multiple concurrent requests"""
        concurrent_requests = 5
        logger.info(f"Testing API with {concurrent_requests} concurrent requests")
        
        # Function to send a request
        def send_request():
            img_data = create_test_image()
            files = {'image_file': ('test.jpg', img_data, 'image/jpeg')}
            start_time = time.time()
            response = requests.post(f"{API_BASE_URL}/detect", files=files)
            elapsed = time.time() - start_time
            return response.status_code, elapsed
        
        # Send concurrent requests
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(send_request) for _ in range(concurrent_requests)]
            results = []
            for future in as_completed(futures):
                status_code, elapsed = future.result()
                results.append((status_code, elapsed))
        
        # Check all responses were successful
        for status_code, elapsed in results:
            self.assertEqual(status_code, 200)
        
        # Log performance statistics
        response_times = [elapsed for _, elapsed in results]
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        logger.info(f"Load test results:")
        logger.info(f"  Average response time: {avg_time:.2f}s")
        logger.info(f"  Minimum response time: {min_time:.2f}s")
        logger.info(f"  Maximum response time: {max_time:.2f}s")


if __name__ == "__main__":
    unittest.main() 