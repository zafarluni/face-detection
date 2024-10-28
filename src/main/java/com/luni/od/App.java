/**
 * YoloObjectDetector is a class designed for detecting objects in images using 
 * the YOLO (You Only Look Once) deep learning model. This class leverages the 
 * OpenCV library to process images and perform object detection efficiently. 
 * 
 * It provides methods to:
 * 1. Initialize the YOLO model with configuration and weights.
 * 2. Detect objects in a given image and return their bounding boxes, class IDs, 
 *    and confidence scores.
 * 3. Annotate detected objects on the image and save the output to a specified 
 *    file path.
 * 4. Validate whether an image contains exactly one human face that occupies a 
 *    specified percentage of the image area. This is particularly useful for 
 *    verifying that profile pictures are valid, ensuring that only one face 
 *    is present and that it occupies a certain percentage of the image.
 * 
 * The detection process includes preprocessing images, running forward passes 
 * through the model, and applying Non-Maximum Suppression to filter out 
 * redundant detections. This class can be used in applications requiring real-time 
 * object detection, such as surveillance, autonomous vehicles, and image 
 * processing tasks.
 * 
 * Author: Zafar Hussain Luni
 */

package com.luni.od;

import java.io.File;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class App {
    private static final Logger logger = LoggerFactory.getLogger(App.class);

    static {
        OpenCVLoader.loadLibrary();
    }

    public static void main(String[] args) {

        String inputFolderPath = "/home/zafarluni/faces/";
        String outputFolderPath = "/home/zafarluni/out/";

        YoloObjectDetectorHelper helper = new YoloObjectDetectorHelper();

        YoloObjectDetector detector = new YoloObjectDetector.Builder(helper.getModelConfiguration(),
                helper.getModelWeight()).setMinFacePercentage(70.0f).build();

        try {
            File inputFolder = new File(inputFolderPath);

            File[] imageFiles = inputFolder.listFiles(
                    (dir, name) -> name.toLowerCase().endsWith(".jpg") || name.toLowerCase().endsWith(".png"));
            if (imageFiles == null || imageFiles.length == 0) {
                logger.error("No image files found in the input folder: {}", inputFolderPath);
                return;
            }

            for (File imageFile : imageFiles) {
                String imagePath = imageFile.getAbsolutePath();
                String outputImagePath = outputFolderPath + imageFile.getName();
                logger.info("Processing image: {}", imagePath);

                // Check if the image is a valid face image
                boolean isValidFaceImage = detector.isValidDisplayPicture(imagePath);
                if (isValidFaceImage) {
                    logger.info("The image {} is valid for Face ID", imagePath);

                    // Detect objects and save output image
                    detector.detectObjectsAndSave(imagePath, outputImagePath, helper.getLabelMap());
                } else {
                    logger.warn("The image {} is not valid for Profile Picture.", imagePath);
                }
            }
        } catch (Exception e) {
            logger.error("An error occurred while processing images.", e);
        }
    }
}
