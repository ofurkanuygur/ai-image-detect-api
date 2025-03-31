import requests
import sys
import os
import json
from pprint import pprint
from datetime import datetime
import argparse
import glob

def test_single_image(image_path, endpoint="detect", verbose=True):
    """
    Test a single image against specified API endpoint
    
    Args:
        image_path: Path to the image
        endpoint: API endpoint to test ("detect", "analysis_details")
        verbose: Whether to print detailed results
    
    Returns:
        API response as dict or None if error
    """
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found!")
        return None
    
    if verbose:
        print(f"\nTesting image: {image_path}")
    
    # Determine content type based on file extension
    content_type = "image/jpeg"  # Default
    if image_path.lower().endswith(".png"):
        content_type = "image/png"
    elif image_path.lower().endswith((".heic", ".heif")):
        content_type = "image/heic"
    
    url = f"http://localhost:8000/api/{endpoint}"
    
    try:
        with open(image_path, 'rb') as img_file:
            files = {'image_file': (os.path.basename(image_path), img_file, content_type)}
            response = requests.post(url, files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                if verbose:
                    print_result_summary(result)
                
                return result
            else:
                print(f"Error: API request failed with status code {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def test_batch(image_paths, verbose=True):
    """
    Test multiple images in a single batch request
    """
    if not image_paths:
        print("Error: No image paths provided for batch testing")
        return None
    
    url = "http://localhost:8000/api/detect/batch"
    
    if verbose:
        print(f"\nBatch testing {len(image_paths)} images")
    
    try:
        files = []
        for i, path in enumerate(image_paths):
            if not os.path.exists(path):
                print(f"Warning: File {path} not found, skipping")
                continue
            
            with open(path, 'rb') as img_file:
                content_type = "image/jpeg"
                if path.lower().endswith(".png"):
                    content_type = "image/png"
                elif path.lower().endswith((".heic", ".heif")):
                    content_type = "image/heic"
                
                files.append(('image_files', (os.path.basename(path), img_file.read(), content_type)))
        
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            results = response.json()
            
            if verbose:
                print("\nBatch Results Summary:")
                for i, result in enumerate(results['results']):
                    print(f"\nImage {i+1}:")
                    print_result_summary(result)
            
            return results
        else:
            print(f"Error: Batch API request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
    
    except Exception as e:
        print(f"Error in batch testing: {str(e)}")
        return None

def compare_images(image_path1, image_path2, verbose=True):
    """
    Compare two images using the comparison endpoint
    """
    url = "http://localhost:8000/api/compare"
    
    if verbose:
        print(f"\nComparing images:")
        print(f"Image 1: {image_path1}")
        print(f"Image 2: {image_path2}")
    
    if not os.path.exists(image_path1) or not os.path.exists(image_path2):
        print("Error: One or both files not found")
        return None
    
    try:
        with open(image_path1, 'rb') as f1, open(image_path2, 'rb') as f2:
            files = {
                'image_file1': (os.path.basename(image_path1), f1, 'image/jpeg'),
                'image_file2': (os.path.basename(image_path2), f2, 'image/jpeg')
            }
            
            response = requests.post(url, files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                if verbose:
                    print("\nComparison Results:")
                    print(f"Summary: {result['summary']}")
                    print("\nImage 1 Analysis:")
                    print_result_summary(result['image1'])
                    print("\nImage 2 Analysis:")
                    print_result_summary(result['image2'])
                
                return result
            else:
                print(f"Error: Comparison API request failed with status code {response.status_code}")
                print(f"Response: {response.text}")
                return None
    
    except Exception as e:
        print(f"Error in image comparison: {str(e)}")
        return None

def health_check():
    """
    Test the health check endpoint
    """
    url = "http://localhost:8000/api/health"
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            result = response.json()
            print("\nHealth Check:")
            print(f"Status: {result['status']}")
            print(f"Version: {result['version']}")
            print(f"Timestamp: {datetime.fromtimestamp(result['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
            return result
        else:
            print(f"Error: Health check failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
    
    except Exception as e:
        print(f"Error in health check: {str(e)}")
        return None

def print_result_summary(result):
    """Helper function to print result summary"""
    print("\nAI Detection Result:")
    print(f"Is AI Generated: {result['is_ai_generated']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    print("\nIndicators:")
    for indicator in result['indicators']:
        print(f"- {indicator}")
    
    print(f"\nImage Format: {result['metadata'].get('format', 'Unknown')}")
    print(f"Image Size: {result['metadata'].get('size', ['Unknown', 'Unknown'])}")
    
    # Print noise and frequency data
    if 'noise_analysis' in result and 'error' not in result['noise_analysis']:
        print(f"Noise Level: {result['noise_analysis']['noise_level']:.2f}")
    
    if 'frequency_analysis' in result and 'error' not in result['frequency_analysis']:
        print(f"High/Low Frequency Ratio: {result['frequency_analysis']['energy_ratio_high_low']:.2f}")

def test_directory(directory_path, save_results=False):
    """
    Test all images in a directory and provide a summary report
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory {directory_path} not found!")
        return
    
    print(f"\nTesting all images in directory: {directory_path}")
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.heic', '*.heif']:
        image_files.extend(glob.glob(os.path.join(directory_path, ext)))
        image_files.extend(glob.glob(os.path.join(directory_path, ext.upper())))
    
    if not image_files:
        print("No image files found in directory")
        return
    
    print(f"Found {len(image_files)} image files")
    
    results = []
    ai_detected_count = 0
    
    for image_path in image_files:
        print(f"\nProcessing {os.path.basename(image_path)}...")
        result = test_single_image(image_path, verbose=False)
        
        if result:
            if result['is_ai_generated']:
                ai_detected_count += 1
                status = "AI DETECTED"
            else:
                status = "REAL IMAGE"
            
            print(f"Result: {status} with confidence {result['confidence']:.2f}")
            results.append({
                'filename': os.path.basename(image_path),
                'result': result
            })
    
    # Print summary
    print("\n===== TEST SUMMARY =====")
    print(f"Total images tested: {len(results)}")
    print(f"AI-generated images detected: {ai_detected_count} ({ai_detected_count/len(results)*100:.1f}%)")
    print(f"Real images detected: {len(results) - ai_detected_count} ({(len(results) - ai_detected_count)/len(results)*100:.1f}%)")
    
    # Save results to file if requested
    if save_results and results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"test_results_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {result_file}")
    
    return results

def main():
    """Main function to run tests based on command-line arguments"""
    parser = argparse.ArgumentParser(description="Test suite for AI Image Detector API")
    
    # Create subparsers for different test commands
    subparsers = parser.add_subparsers(dest="command", help="Test command")
    
    # Single image test
    single_parser = subparsers.add_parser("test", help="Test a single image")
    single_parser.add_argument("image_path", help="Path to the image file")
    single_parser.add_argument("--detailed", action="store_true", help="Get detailed analysis")
    
    # Batch test
    batch_parser = subparsers.add_parser("batch", help="Test multiple images in batch")
    batch_parser.add_argument("image_paths", nargs="+", help="Paths to image files")
    
    # Directory test
    dir_parser = subparsers.add_parser("directory", help="Test all images in a directory")
    dir_parser.add_argument("directory_path", help="Path to directory with images")
    dir_parser.add_argument("--save", action="store_true", help="Save results to JSON file")
    
    # Compare images
    compare_parser = subparsers.add_parser("compare", help="Compare two images")
    compare_parser.add_argument("image_path1", help="Path to first image file")
    compare_parser.add_argument("image_path2", help="Path to second image file")
    
    # Health check
    subparsers.add_parser("health", help="Check API health")
    
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "test":
        endpoint = "analysis_details" if args.detailed else "detect"
        test_single_image(args.image_path, endpoint=endpoint)
    
    elif args.command == "batch":
        test_batch(args.image_paths)
    
    elif args.command == "directory":
        test_directory(args.directory_path, save_results=args.save)
    
    elif args.command == "compare":
        compare_images(args.image_path1, args.image_path2)
    
    elif args.command == "health":
        health_check()
    
    else:
        # If no command provided, print help
        parser.print_help()

if __name__ == "__main__":
    main() 