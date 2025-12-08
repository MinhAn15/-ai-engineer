"""
Image preprocessing to improve OCR accuracy
"""
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np


def preprocess_for_ocr(image: Image.Image, method: str = 'adaptive') -> Image.Image:
    '''
    Preprocessing image to improve OCR accuracy.
    
    Args:
        image: PIL Image object
        method: 'adaptive', 'simple', 'advanced'
    
    Returns:
        Preprocessed PIL Image
    '''
    # Convert to numpy array for OpenCV processing
    img_array = np.array(image)
    
    if method == 'simple':
        return preprocess_simple(image)
    elif method == 'adaptive':
        return preprocess_adaptive(img_array)
    elif method == 'advanced':
        return preprocess_advanced(img_array)
    else:
        return image


def preprocess_simple(image: Image.Image) -> Image.Image:
    '''Simple PIL-based preprocessing'''
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # Sharpen
    image = image.filter(ImageFilter.SHARPEN)
    
    # Binarization (black & white)
    image = ImageOps.autocontrast(image)
    
    return image


def preprocess_adaptive(img_array: np.ndarray) -> Image.Image:
    '''Adaptive thresholding with OpenCV (BEST for varied lighting)'''
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        denoised, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        2
    )
    
    # Convert back to PIL
    return Image.fromarray(binary)


def preprocess_advanced(img_array: np.ndarray) -> Image.Image:
    '''Advanced preprocessing with deskewing and morphology'''
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Deskew (fix rotation)
    coords = np.column_stack(np.where(denoised > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        (h, w) = denoised.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            denoised, M, (w, h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_REPLICATE
        )
    else:
        rotated = denoised
    
    # Morphological operations (remove small noise)
    kernel = np.ones((1, 1), np.uint8)
    morph = cv2.morphologyEx(rotated, cv2.MORPH_CLOSE, kernel)
    
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        morph, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        2
    )
    
    return Image.fromarray(binary)


if __name__ == '__main__':
    # Test preprocessing
    import sys
    
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        test_path = 'image/invoice.jpg'
    
    print(f"Testing preprocessing on: {test_path}")
    test_img = Image.open(test_path)
    
    simple = preprocess_simple(test_img.copy())
    simple.save('outputs/preprocessed_simple.png')
    print("✓ Simple preprocessing saved")
    
    adaptive = preprocess_for_ocr(test_img.copy(), 'adaptive')
    adaptive.save('outputs/preprocessed_adaptive.png')
    print("✓ Adaptive preprocessing saved")
    
    advanced = preprocess_for_ocr(test_img.copy(), 'advanced')
    advanced.save('outputs/preprocessed_advanced.png')
    print("✓ Advanced preprocessing saved")
    
    print('\n✓ All preprocessed images saved to outputs/')
