import requests
import sys
import os
from pprint import pprint

def test_detector(image_path):
    """Test the AI detector with a given image file"""
    print(f"\nTesting image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found!")
        return
    
    # API endpoint
    url = "http://localhost:8000/api/detect"
    
    try:
        with open(image_path, 'rb') as img_file:
            # Prepare multipart/form-data
            files = {'image_file': (os.path.basename(image_path), img_file, 'image/jpeg')}
            
            # Send request
            response = requests.post(url, files=files)
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                
                # Print summary of results
                print("\nAI Detection Result:")
                print(f"Is AI Generated: {result['is_ai_generated']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print("\nIndicators:")
                for indicator in result['indicators']:
                    print(f"- {indicator}")
                
                # Print metadata format for debugging
                print(f"\nImage Format: {result['metadata'].get('format', 'Unknown')}")
                
                # Return the result
                return result
            else:
                print(f"Error: API request failed with status code {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Check if image path is provided as command-line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_detector(image_path)
    else:
        print("Usage: python test_detector.py <path_to_image>")
        print("Example: python test_detector.py test_images/photo.jpg") 