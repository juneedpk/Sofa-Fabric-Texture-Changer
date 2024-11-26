import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def remove_background(image):
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    image_array = np.array(image)
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    mask = ((s > 20) | (v < 240)).astype(np.uint8) * 255
    
    # Enhanced smoothing process
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Multiple passes of Gaussian blur
    mask_float = mask.astype(np.float32) / 255.0
    mask_float = cv2.GaussianBlur(mask_float, (21, 21), 0)
    mask_float = cv2.GaussianBlur(mask_float, (21, 21), 0)
    
    # Create feathered edges
    mask_float = cv2.normalize(mask_float, None, 0, 1, cv2.NORM_MINMAX)
    
    white_bg = np.ones_like(image_array) * 255
    mask_4channel = np.stack([mask_float] * 4, axis=-1)
    
    result = image_array * mask_4channel + white_bg * (1 - mask_4channel)
    return Image.fromarray(result.astype(np.uint8))

def enhance_texture_resolution(image, scale_factor=2.0):
    """Enhance the resolution of the texture while preserving details"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    # Calculate new dimensions
    new_width = int(image.size[0] * scale_factor)
    new_height = int(image.size[1] * scale_factor)
    
    # Use high-quality upscaling
    enhanced = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return np.array(enhanced)

def normalize_dpi(image, target_dpi=300):
    """Normalize image to specified DPI while preserving physical size"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    # Get current DPI if available
    try:
        current_dpi = image.info.get('dpi', (target_dpi, target_dpi))[0]
    except:
        current_dpi = target_dpi
    
    if current_dpi != target_dpi:
        # Calculate scaling factor
        scale_factor = target_dpi / current_dpi
        new_size = tuple(int(dim * scale_factor) for dim in image.size)
        
        # Resize image with high-quality resampling
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Set new DPI
        image.info['dpi'] = (target_dpi, target_dpi)
    
    return image

def scale_fabric_texture(fabric_image, scale_percent):
    """Scale the fabric texture pattern by a percentage with enhanced resolution"""
    # Convert to numpy array if needed
    if isinstance(fabric_image, Image.Image):
        fabric_array = np.array(fabric_image)
    else:
        fabric_array = fabric_image
    
    # Ensure proper color handling
    if len(fabric_array.shape) == 2:
        fabric_array = cv2.cvtColor(fabric_array, cv2.COLOR_GRAY2RGB)
    elif fabric_array.shape[2] == 4:
        fabric_array = cv2.cvtColor(fabric_array, cv2.COLOR_RGBA2RGB)
    
    # Enhance texture resolution first
    fabric_array = enhance_texture_resolution(fabric_array, scale_factor=2.0)
    
    # Calculate new dimensions
    scale_factor = scale_percent / 100.0
    original_height, original_width = fabric_array.shape[:2]
    
    # Calculate new size while maintaining aspect ratio
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # Use high-quality resizing for the fabric texture
    resized_fabric = cv2.resize(fabric_array, (new_width, new_height), 
                              interpolation=cv2.INTER_LANCZOS4)
    
    # Create a repeating pattern to fill the original size
    pattern_height = original_height
    pattern_width = original_width
    
    # Create empty array for the pattern
    pattern = np.zeros((pattern_height, pattern_width, 3), dtype=np.uint8)
    
    # Fill the pattern by tiling the resized fabric
    for y in range(0, pattern_height, new_height):
        for x in range(0, pattern_width, new_width):
            y_end = min(y + new_height, pattern_height)
            x_end = min(x + new_width, pattern_width)
            y_slice = slice(y, y_end)
            x_slice = slice(x, x_end)
            
            pattern_slice_height = y_end - y
            pattern_slice_width = x_end - x
            
            pattern[y_slice, x_slice] = resized_fabric[:pattern_slice_height, :pattern_slice_width]
    
    return pattern

def apply_fabric_to_sofa(sofa_image, fabric_image, fabric_scale=100, target_dpi=300):
    """
    Apply high-resolution fabric texture to sofa
    """
    # Normalize DPI for both images at high resolution
    sofa_image = normalize_dpi(sofa_image, target_dpi)
    fabric_image = normalize_dpi(fabric_image, target_dpi)
    
    # Convert PIL images to numpy arrays with proper color handling
    sofa_array = np.array(sofa_image)
    if len(sofa_array.shape) == 2:
        sofa_array = cv2.cvtColor(sofa_array, cv2.COLOR_GRAY2RGB)
    elif sofa_array.shape[2] == 4:
        sofa_array = cv2.cvtColor(sofa_array, cv2.COLOR_RGBA2RGB)
    
    fabric_array = np.array(fabric_image)
    if len(fabric_array.shape) == 2:
        fabric_array = cv2.cvtColor(fabric_array, cv2.COLOR_GRAY2RGB)
    elif fabric_array.shape[2] == 4:
        fabric_array = cv2.cvtColor(fabric_array, cv2.COLOR_RGBA2RGB)
    
    # Scale and enhance the fabric texture
    scaled_fabric = scale_fabric_texture(fabric_array, fabric_scale)
    
    # Convert images to HSV color space for better segmentation
    sofa_hsv = cv2.cvtColor(sofa_array, cv2.COLOR_RGB2HSV)
    
    # Create multiple masks for different parts of the sofa
    light_lower = np.array([0, 0, 180])
    light_upper = np.array([180, 30, 255])
    light_mask = cv2.inRange(sofa_hsv, light_lower, light_upper)
    
    mid_lower = np.array([0, 0, 100])
    mid_upper = np.array([180, 60, 180])
    mid_mask = cv2.inRange(sofa_hsv, mid_lower, mid_upper)
    
    # Combine masks
    mask = cv2.bitwise_or(light_mask, mid_mask)
    
    # Clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
        
        # Create perspective transform points
        hull = cv2.convexHull(largest_contour)
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Sort box points
        box = box[box[:,0].argsort()]
        left = box[:2]
        right = box[2:]
        left = left[left[:,1].argsort()]
        right = right[right[:,1].argsort()]
        
        # Get source and destination points for perspective transform
        src_points = np.float32([[0, 0], [scaled_fabric.shape[1], 0],
                                [0, scaled_fabric.shape[0]], [scaled_fabric.shape[1], scaled_fabric.shape[0]]])
        dst_points = np.float32([left[0], right[0], left[1], right[1]])
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Warp fabric to match sofa perspective
        warped_fabric = cv2.warpPerspective(scaled_fabric, matrix, (sofa_array.shape[1], sofa_array.shape[0]))
        
        # Create output image with white background
        output = np.ones_like(sofa_array) * 255
        
        # Convert mask to float and apply Gaussian blur for smooth edges
        mask_float = mask.astype(np.float32) / 255.0
        mask_float = cv2.GaussianBlur(mask_float, (21, 21), 0)
        mask_float = cv2.GaussianBlur(mask_float, (21, 21), 0)
        
        # Convert mask to 3 channels
        mask_3channel = np.stack([mask_float] * 3, axis=-1)
        
        # Extract lighting information from original sofa
        sofa_gray = cv2.cvtColor(sofa_array, cv2.COLOR_RGB2GRAY)
        lighting = sofa_gray.astype(float) / 255.0
        
        # Apply lighting to fabric with increased brightness
        warped_fabric = warped_fabric.astype(float)
        lighting = lighting * 1.2  # Increase overall brightness
        for c in range(3):
            warped_fabric[:,:,c] *= lighting
        
        # Add shadows and highlights with increased brightness
        shadows = np.minimum(lighting, 0.8)  # Increased from 0.7 to 0.8
        highlights = np.maximum(lighting, 0.4)  # Increased from 0.3 to 0.4
        warped_fabric *= shadows[:,:,np.newaxis]
        warped_fabric += (highlights[:,:,np.newaxis] * 0.4)  # Increased highlight intensity
        
        # Clip values to valid range
        warped_fabric = np.clip(warped_fabric, 0, 255).astype(np.uint8)
        
        # Blend warped fabric with white background using mask
        output = output * (1 - mask_3channel) + warped_fabric * mask_3channel
        output = output.astype(np.uint8)
        
        # Convert to PIL Image
        result_image = Image.fromarray(output)
        
        # Ensure output maintains high DPI
        result_image.info['dpi'] = (target_dpi, target_dpi)
        
        return result_image
    
    return sofa_image

# Example usage:
if __name__ == "__main__":
    # Open sofa image
    sofa_image = Image.open('sofa_image.jpg').convert('RGB')

    # Open fabric image
    fabric_image = Image.open('fabric_image.jpg').convert('RGB')

    # Set parameters for high-resolution output
    fabric_scale = 100  # Adjust pattern size as needed
    target_dpi = 300    # High DPI for better detail

    # Process image with enhanced resolution
    result = apply_fabric_to_sofa(sofa_image, fabric_image, fabric_scale, target_dpi)

    # Display result
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(sofa_image)
    plt.axis('off')
    plt.title('Original Sofa')

    plt.subplot(1, 3, 2)
    plt.imshow(fabric_image)
    plt.axis('off')
    plt.title('Fabric Texture')

    plt.subplot(1, 3, 3)
    plt.imshow(result)
    plt.axis('off')
    plt.title('Result')
    plt.show()

    # Save high-resolution result
    result.save('retextured_sofa_hires.png', dpi=(target_dpi, target_dpi), quality=100)
