import cv2
import os
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt

load_dotenv()

project_path = os.getenv("Project_PATH")
if not project_path:
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

cat_input_dir = os.path.join(project_path, "chest_xray", "train", "NORMAL")
dog_input_dir = os.path.join(project_path, "chest_xray", "train", "PNEUMONIA")
cat_output_dir = os.path.join(project_path, "processed", "train", "NORMAL")
dog_output_dir = os.path.join(project_path, "processed", "train", "PNEUMONIA")

os.makedirs(cat_output_dir, exist_ok=True)
os.makedirs(dog_output_dir, exist_ok=True)

def preprocess(dir, output_dir):
    for img_name in os.listdir(dir):
        path = os.path.join(dir, img_name)
        im = cv2.imread(path)
        if im is None:
            print(f"Warning: Could not read image {path}, skipping.")
            continue
        # cv2.imshow("Original Image", im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        im = cv2.resize(im, (256, 256),1,1,interpolation=cv2.INTER_CUBIC)
        # cv2.imshow("Resized Image", im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Grayscale Image", im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # im = cv2.GaussianBlur(im,(3,3),0)
        im = cv2.medianBlur(im, 3)
        # cv2.imshow("Blurred Image", im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        im = cv2.filter2D(im, -1, kernel)
        # cv2.imshow("Sharpened Image", im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        Hist = cv2.calcHist([im], [0], None, [256], [0, 256])
        cdf = np.cumsum(Hist)
        cdf_normalized = cdf * Hist.max() / cdf.max()
        # plt.subplot(232)
        # plt.imshow(im, cmap='gray')
        # plt.title('Image')
        # plt.subplot(234)
        # plt.plot(Hist)
        # plt.plot(cdf_normalized, color='b')
        # plt.xlabel('Pixel Intensity')
        # plt.ylabel('no of px')
        # plt.show()
        
        claheObj = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        img = claheObj.apply(im)
        # cv2.imshow("CLAHE Image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        claheHist = cv2.calcHist([img], [0], None, [256], [0, 256])
        clahecdf = np.cumsum(claheHist)
        clahecdf_normalized = clahecdf * claheHist.max() / clahecdf.max()
        # plt.subplot(232)
        # plt.imshow(img, cmap='gray')
        # plt.title('CLAHE Image')
        # plt.subplot(234)
        # plt.plot(claheHist)
        # plt.plot(clahecdf_normalized, color='b')
        # plt.xlabel('Pixel Intensity')
        # plt.ylabel('no of px')
        # plt.show()
        
        # normalization
        img = img / 255.0
        
        # save
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, (img * 255).astype(np.uint8))
        
preprocess(cat_input_dir, cat_output_dir)
preprocess(dog_input_dir, dog_output_dir)