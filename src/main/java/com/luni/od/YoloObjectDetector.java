package com.luni.od;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * YoloObjectDetector is a class to detect objects in images using the YOLO (You
 * Only Look Once) object detection model.
 * It provides methods to detect objects and save the annotated images.
 */
public class YoloObjectDetector {
    private static final Logger logger = LoggerFactory.getLogger(YoloObjectDetector.class);
    private Net net;
    private float confidenceThreshold;
    private float nmsThreshold;
    private float minFacePercentage;
    private float maxFacePercentage;

    /**
     * Private constructor for the YoloObjectDetector.
     * 
     * @param builder Builder object used to configure the detector.
     */
    private YoloObjectDetector(Builder builder) {
        logger.info("Initializing YoloObjectDetector with provided configurations.");

        try {

            this.net = Dnn.readNetFromDarknet(builder.modelConfiguration, builder.modelWeights);
        } catch (UnsatisfiedLinkError e) {
            logger.error(
                    "Error initializing YoloObjectDetector. Please ensure OpenCV is properly installed and configured.",
                    e);
            throw new IllegalStateException("Failed to initialize YoloObjectDetector: " + e.getMessage(), e);
        } catch (org.opencv.core.CvException e) {
            logger.error(
                    "Error initializing YoloObjectDetector. Please ensure that the model configuration and weights are available in the application path.",
                    e);
            throw new IllegalStateException("Failed to initialize YoloObjectDetector: " + e.getMessage(), e);

        }

        this.net.setPreferableBackend(builder.preferableBackend);
        this.net.setPreferableTarget(builder.preferableTarget);
        this.confidenceThreshold = builder.confidenceThreshold;
        this.nmsThreshold = builder.nmsThreshold;
        this.minFacePercentage = builder.minFacePercentage;
        this.maxFacePercentage = builder.maxFacePercentage;
    }

    /**
     * Builder class for creating instances of YoloObjectDetector with customized
     * settings.
     */
    public static class Builder {
        private String modelWeights;
        private String modelConfiguration;
        private int preferableBackend = Dnn.DNN_BACKEND_OPENCV;
        private int preferableTarget = Dnn.DNN_TARGET_CPU;
        private float confidenceThreshold = 0.5f;
        private float nmsThreshold = 0.4f;
        private float minFacePercentage = 70.0f;
        private float maxFacePercentage = 90.0f;

        /**
         * Constructor for Builder class.
         * 
         * @param modelConfiguration Path to YOLO model configuration.
         * @param modelWeights       Path to YOLO model weights.
         */
        public Builder(String modelConfiguration, String modelWeights) {
            if (modelConfiguration == null || modelConfiguration.isEmpty() ||
                    modelWeights == null || modelWeights.isEmpty()) {
                throw new IllegalArgumentException(
                        "Model configuration and weights must be provided and cannot be empty");
            }
            this.modelConfiguration = modelConfiguration;
            this.modelWeights = modelWeights;
        }

        /**
         * Set the preferable backend for inference.
         * 
         * @param preferableBackend Backend type (e.g., DNN_BACKEND_OPENCV).
         * @return Builder instance for chaining.
         */
        public Builder setPreferableBackend(int preferableBackend) {
            this.preferableBackend = preferableBackend;
            return this;
        }

        /**
         * Set the preferable target for inference.
         * 
         * @param preferableTarget Target type (e.g., DNN_TARGET_CPU).
         * @return Builder instance for chaining.
         */
        public Builder setPreferableTarget(int preferableTarget) {
            this.preferableTarget = preferableTarget;
            return this;
        }

        /**
         * Set the confidence threshold for detecting objects.
         * 
         * @param confidenceThreshold Confidence threshold to filter weak detections.
         * @return Builder instance for chaining.
         */
        public Builder setConfidenceThreshold(float confidenceThreshold) {
            this.confidenceThreshold = confidenceThreshold;
            return this;
        }

        /**
         * Set the Non-Maximum Suppression (NMS) threshold.
         * 
         * @param nmsThreshold NMS threshold for suppressing overlapping bounding boxes.
         * @return Builder instance for chaining.
         */
        public Builder setNmsThreshold(float nmsThreshold) {
            this.nmsThreshold = nmsThreshold;
            return this;
        }

        /**
         * Set the minimum percentage of the image that must be occupied by the face.
         * 
         * @param minFacePercentage Minimum percentage for face area.
         * @return Builder instance for chaining.
         */
        public Builder setMinFacePercentage(float minFacePercentage) {
            this.minFacePercentage = minFacePercentage;
            return this;
        }

        /**
         * Set the maximum percentage of the image that must be occupied by the face.
         * 
         * @param maxFacePercentage Maximum percentage for face area.
         * @return Builder instance for chaining.
         */
        public Builder setMaxFacePercentage(float maxFacePercentage) {
            this.maxFacePercentage = maxFacePercentage;
            return this;
        }

        /**
         * Build and return a YoloObjectDetector instance.
         * 
         * @return Configured YoloObjectDetector instance.
         */
        public YoloObjectDetector build() {
            return new YoloObjectDetector(this);
        }
    }

    /**
     * Detect objects in the provided image using the YOLO model.
     * 
     * @param image Input image as a Mat object.
     * @return List of detected objects, each represented by a map containing
     *         bounding box, classId, and confidence.
     */
    private List<Map<String, Object>> detect(Mat image) {
        logger.debug("Starting object detection.");
        Mat blob = preprocessImage(image);
        List<Mat> result = runForwardPass(blob);
        return processOutput(result, image);
    }

    /**
     * Preprocess the input image for YOLO.
     * 
     * @param image Input image as a Mat object.
     * @return Preprocessed blob for YOLO.
     */
    private Mat preprocessImage(Mat image) {
        logger.debug("Preprocessing image for YOLO.");
        return Dnn.blobFromImage(image, 1 / 255.0, new Size(416, 416), new Scalar(0, 0, 0), true, false);
    }

    /**
     * Run forward pass to get predictions from the YOLO model.
     * 
     * @param blob Preprocessed input blob.
     * @return List of Mat objects containing the predictions.
     */
    private List<Mat> runForwardPass(Mat blob) {
        logger.debug("Running forward pass on the model.");
        net.setInput(blob);
        List<String> outBlobNames = net.getUnconnectedOutLayersNames();
        List<Mat> result = new ArrayList<>();
        net.forward(result, outBlobNames);
        return result;
    }

    /**
     * Process the output of the forward pass to extract detected objects.
     * 
     * @param result List of Mat objects containing predictions.
     * @param image  Original input image.
     * @return List of detected objects, each represented by a map containing
     *         bounding box, classId, and confidence.
     */
    private List<Map<String, Object>> processOutput(List<Mat> result, Mat image) {
        logger.debug("Processing model output to extract detected objects.");
        List<Rect2d> boxes = new ArrayList<>();
        List<Float> confidences = new ArrayList<>();
        List<Integer> classIds = new ArrayList<>();

        for (Mat mat : result) {
            for (int i = 0; i < mat.rows(); i++) {
                Mat row = mat.row(i);
                Mat scores = row.colRange(5, mat.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float) mm.maxVal;
                Point classIdPoint = mm.maxLoc;

                if (confidence > confidenceThreshold) {
                    // Get the bounding box coordinates
                    float centerX = (float) (row.get(0, 0)[0] * image.cols());
                    float centerY = (float) (row.get(0, 1)[0] * image.rows());
                    float width = (float) (row.get(0, 2)[0] * image.cols());
                    float height = (float) (row.get(0, 3)[0] * image.rows());

                    // Convert from center to top-left corner format
                    int x = (int) (centerX - width / 2);
                    int y = (int) (centerY - height / 2);

                    boxes.add(new Rect2d(x, y, width, height));
                    confidences.add(confidence);
                    classIds.add((int) classIdPoint.x);
                }
            }
        }

        // Apply Non-Maximum Suppression (NMS)
        MatOfRect2d matOfBoxes = new MatOfRect2d();
        matOfBoxes.fromList(boxes);
        float[] confidenceArray = new float[confidences.size()];
        for (int i = 0; i < confidences.size(); i++) {
            confidenceArray[i] = confidences.get(i);
        }
        MatOfFloat matOfConfidences = new MatOfFloat(confidenceArray);
        MatOfInt indices = new MatOfInt();

        if (matOfBoxes.empty() || matOfConfidences.empty()) {
            logger.warn("Image does not contain any detections to process.");
            return new ArrayList<>();
        }

        // Non-Maximum Suppression is applied to remove redundant overlapping bounding
        // boxes
        // by keeping only the ones with the highest confidence score.
        Dnn.NMSBoxes(matOfBoxes, matOfConfidences, confidenceThreshold, nmsThreshold, indices);

        // Filter out the final boxes after NMS
        List<Map<String, Object>> finalDetections = new ArrayList<>();
        int[] indicesArray = indices.toArray();

        for (int i : indicesArray) {
            Map<String, Object> detection = new HashMap<>();
            detection.put("box", boxes.get(i));
            detection.put("classId", classIds.get(i));
            detection.put("confidence", confidences.get(i));
            finalDetections.add(detection);
        }

        logger.info("Detected {} objects.", finalDetections.size());
        return finalDetections;
    }

    /**
     * Detect objects in an image and return the list of detected objects.
     * 
     * @param imagePath Path to the input image.
     * @return List of maps, each representing a detected object with bounding box,
     *         classId, and confidence.
     */
    public List<Map<String, Object>> detectObjects(String imagePath) {
        logger.info("Detecting objects in image: {}", imagePath);
        Mat image = Imgcodecs.imread(imagePath);
        if (image.empty()) {
            logger.error("Error: Could not load image from path: {}", imagePath);
            throw new IllegalArgumentException("Error: Could not load image from path: " + imagePath);
        }
        return detect(image);
    }

    /**
     * Detect objects in an image, draw bounding boxes around detected objects, and
     * save the output image.
     * 
     * @param imagePath       Path to the input image.
     * @param outputImagePath Path to save the output image with annotated bounding
     *                        boxes.
     * @param labelMap        Map containing class IDs and their corresponding
     *                        labels.
     */
    public void detectObjectsAndSave(String imagePath, String outputImagePath, Map<Integer, String> labelMap) {
        logger.info("Detecting objects and saving output image. Input: {}, Output: {}", imagePath, outputImagePath);
        Mat image = Imgcodecs.imread(imagePath);
        if (image.empty()) {
            logger.error("Error: Could not load image from path: {}", imagePath);
            throw new IllegalArgumentException("Error: Could not load image from path: " + imagePath);
        }
        List<Map<String, Object>> detections = detect(image);

        // Draw the final bounding boxes with labels from labelMap
        drawBoundingBoxes(image, detections, labelMap);

        // Save the result image with bounding boxes drawn
        boolean success = Imgcodecs.imwrite(outputImagePath, image);
        if (!success) {
            logger.error("Error: Failed to save the output image. Please check the file path and permissions.");
        } else {
            logger.info("Output image saved successfully at: {}", outputImagePath);
        }
    }

    /**
     * Draw bounding boxes around detected objects in an image.
     * 
     * @param image      Input image as a Mat object.
     * @param detections List of detected objects.
     * @param labelMap   Map containing class IDs and their corresponding labels.
     */
    private void drawBoundingBoxes(Mat image, List<Map<String, Object>> detections, Map<Integer, String> labelMap) {
        logger.debug("Drawing bounding boxes on detected objects.");
        for (Map<String, Object> detection : detections) {
            Rect2d box = (Rect2d) detection.get("box");
            Imgproc.rectangle(image, new Point(box.x, box.y), new Point(box.x + box.width, box.y + box.height),
                    new Scalar(0, 255, 0), 2);
            int classId = (int) detection.get("classId");
            String label = labelMap.getOrDefault(classId, "Unknown") + " Confidence: "
                    + String.format("%.2f", detection.get("confidence"));
            Imgproc.putText(image, label, new Point(box.x, box.y - 5), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5,
                    new Scalar(0, 255, 0), 1);
        }
    }

    /**
     * Check if an image contains exactly one face that occupies a configurable
     * percentage of the image
     * area and ensure the detected face is of a human.
     * 
     * @param imagePath Path to the input image.
     * @return true if the image is considered a valid human face image; false
     *         otherwise.
     */
    public boolean isValidDisplayPicture(String imagePath) {
        logger.info("Checking if the image is a valid face image: {}", imagePath);
        Mat image = Imgcodecs.imread(imagePath);
        if (image.empty()) {
            logger.error("Error: Could not load image from path: {}", imagePath);
            throw new IllegalArgumentException("Error: Could not load image from path: " + imagePath);
        }
        List<Map<String, Object>> detections = detect(image);

        // Check if there is exactly one bounding box
        if (detections.size() == 1) {
            Rect2d box = (Rect2d) detections.get(0).get("box");
            double faceArea = box.width * box.height;
            double imageArea = image.cols() * image.rows();
            double facePercentage = (faceArea / imageArea) * 100;

            int classId = (int) detections.get(0).get("classId");

            // Check if face constitutes the configured percentage of the image and is a
            // human face
            if (facePercentage >= minFacePercentage && facePercentage <= maxFacePercentage && classId == 0) {
                logger.info(
                        "The detected face occupies {}% of the image, which is within the acceptable range ({}% - {}).",
                        facePercentage, minFacePercentage, maxFacePercentage);
                return true;
            } else {
                logger.warn(
                        "The face does not occupy the required {}% to {}% of the image or is not a human face. Detected face occupies {}% of the image with classId {}.",
                        minFacePercentage, maxFacePercentage, facePercentage, classId);
                return false;
            }
        } else {
            logger.warn("Invalid number of faces detected. Expected 1 face but found {}.", detections.size());
            return false;
        }
    }
}
