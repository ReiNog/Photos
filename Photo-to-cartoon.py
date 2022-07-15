import cv2
import numpy as np

img_path = 'D:\Dev\Photos\img\Selfie-202207-avatar.jpg'

def read_file(filename):
  img = cv2.imread(filename)
  return img

def show_photo(img, window_name):
    cv2.imshow(window_name, img)
    # Waits for user to press any key. This is necessary to avoid Python kernel form crashing.
    cv2.waitKey(0) 
    # Closing all open windows 
    cv2.destroyAllWindows() 

def edge_mask(img, line_size, blur_value):
    # A cartoon effect emphasizes the thickness of the edge in an image. To do that, we transform the image in grayscale, reduce the noise and define the line size of the edge.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges

def color_quantization(img, k):
    # Transform the image
    data = np.float32(img).reshape((-1, 3))

    # Determine criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # Implementing K-Means
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

# Upload and show the original photo
orig_photo = read_file(img_path)
show_photo(orig_photo, 'Foto Original')

# Create edge mask and show edges
line_size = 9
blur_value = 9
edges = edge_mask(orig_photo, line_size, blur_value)
show_photo(edges, 'Mascara de Contorno')

# Redude the collor palette.
total_color = 12 
quant_photo = color_quantization(orig_photo, total_color)
show_photo(quant_photo, 'Foto Simplificada')

# Billateral filter to reduce the noise in the image. It gives a bit blurred and sharpness-reducing effect to the image.
#    d — Diameter of each pixel neighborhood
#    sigmaColor — A larger value of the parameter means larger areas of semi-equal color.
#    sigmaSpace –A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough.
blurred_photo = cv2.bilateralFilter(quant_photo, d=5, sigmaColor=200,sigmaSpace=200)
show_photo(blurred_photo, 'Foto Desfocada')

# Cartoon is the combination of the edgde mask and the blurred photo.
cartoon = cv2.bitwise_and(blurred_photo, blurred_photo, mask=edges)
show_photo(cartoon, "Cartoon")
cv2.imwrite('D:\Dev\Photos\img\Rei-Cartoon.jpg', cartoon)
