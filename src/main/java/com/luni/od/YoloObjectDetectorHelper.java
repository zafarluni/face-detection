/**
 * YoloObjectDetectorHelper is a utility class that assists in managing the 
 * resources required for the YOLO (You Only Look Once) object detection model. 
 * This class is responsible for loading the model weights and configuration files, 
 * as well as initializing a label map that associates class IDs with their 
 * corresponding object names.
 * 
 * The class provides methods to:
 * 1. Retrieve the paths to the model weights and configuration files by loading 
 *    them into temporary files from the resources.
 * 2. Access the label map, which contains a mapping of class IDs to object names 
 *    that YOLO can detect.
 * 
 * This helper class streamlines the setup process for the YOLO object detector, 
 * ensuring that all necessary resources are readily available for efficient 
 * object detection in images.
 * 
 * Author: Zafar Hussain Luni
 */

package com.luni.od;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

public class YoloObjectDetectorHelper {
    private String modelWeights;
    private String modelConfiguration;
    private Map<Integer, String> labelMap;

    public YoloObjectDetectorHelper() {
        this.modelWeights = PropertiesUtil.get().getProperty("yolov3.weights.path");
        this.modelConfiguration = PropertiesUtil.get().getProperty("yolov3.configuration.path");
        this.labelMap = initializeLabelMap();
    }

    public String getModelWeight() {
        return loadResourceToTempFile(modelWeights);
    }

    public String getModelConfiguration() {
        return loadResourceToTempFile(modelConfiguration);
    }

    public Map<Integer, String> getLabelMap() {
        return this.labelMap;
    }

    private String loadResourceToTempFile(String resourcePath) {
        try (InputStream resourceStream = YoloObjectDetectorHelper.class.getClassLoader()
                .getResourceAsStream(resourcePath)) {
            if (resourceStream == null) {
                throw new IllegalArgumentException("Resource not found: " + resourcePath);
            }
            // Create a temporary file
            Path tempFile = Files.createTempFile("temp", null);
            // Copy the input stream to the temporary file
            Files.copy(resourceStream, tempFile, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            // Delete the temporary file on exit
            tempFile.toFile().deleteOnExit();
            return tempFile.toString(); // Return the path to the temporary file
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private Map<Integer, String> initializeLabelMap() {
        Map<Integer, String> labelMap = new HashMap<>();
        labelMap.put(0, "person");
        labelMap.put(1, "bicycle");
        labelMap.put(2, "car");
        labelMap.put(3, "motorbike");
        labelMap.put(4, "aeroplane");
        labelMap.put(5, "bus");
        labelMap.put(6, "train");
        labelMap.put(7, "truck");
        labelMap.put(8, "boat");
        labelMap.put(9, "traffic light");
        labelMap.put(10, "fire hydrant");
        labelMap.put(11, "stop sign");
        labelMap.put(12, "parking meter");
        labelMap.put(13, "bench");
        labelMap.put(14, "bird");
        labelMap.put(15, "cat");
        labelMap.put(16, "dog");
        labelMap.put(17, "horse");
        labelMap.put(18, "sheep");
        labelMap.put(19, "cow");
        labelMap.put(20, "elephant");
        labelMap.put(21, "bear");
        labelMap.put(22, "zebra");
        labelMap.put(23, "giraffe");
        labelMap.put(24, "backpack");
        labelMap.put(25, "umbrella");
        labelMap.put(26, "handbag");
        labelMap.put(27, "tie");
        labelMap.put(28, "suitcase");
        labelMap.put(29, "frisbee");
        labelMap.put(30, "skis");
        labelMap.put(31, "snowboard");
        labelMap.put(32, "sports ball");
        labelMap.put(33, "kite");
        labelMap.put(34, "baseball bat");
        labelMap.put(35, "baseball glove");
        labelMap.put(36, "skateboard");
        labelMap.put(37, "surfboard");
        labelMap.put(38, "tennis racket");
        labelMap.put(39, "bottle");
        labelMap.put(40, "wine glass");
        labelMap.put(41, "cup");
        labelMap.put(42, "fork");
        labelMap.put(43, "knife");
        labelMap.put(44, "spoon");
        labelMap.put(45, "bowl");
        labelMap.put(46, "banana");
        labelMap.put(47, "apple");
        labelMap.put(48, "sandwich");
        labelMap.put(49, "orange");
        labelMap.put(50, "broccoli");
        labelMap.put(51, "carrot");
        labelMap.put(52, "hot dog");
        labelMap.put(53, "pizza");
        labelMap.put(54, "donut");
        labelMap.put(55, "cake");
        labelMap.put(56, "chair");
        labelMap.put(57, "couch");
        labelMap.put(58, "potted plant");
        labelMap.put(59, "bed");
        labelMap.put(60, "dining table");
        labelMap.put(61, "toilet");
        labelMap.put(62, "TV");
        labelMap.put(63, "laptop");
        labelMap.put(64, "mouse");
        labelMap.put(65, "remote");
        labelMap.put(66, "keyboard");
        labelMap.put(67, "cell phone");
        labelMap.put(68, "microwave");
        labelMap.put(69, "oven");
        labelMap.put(70, "toaster");
        labelMap.put(71, "sink");
        labelMap.put(72, "refrigerator");
        labelMap.put(73, "book");
        labelMap.put(74, "clock");
        labelMap.put(75, "vase");
        labelMap.put(76, "scissors");
        labelMap.put(77, "teddy bear");
        labelMap.put(78, "hair drier");
        labelMap.put(79, "toothbrush");

        return labelMap;
    }
}
