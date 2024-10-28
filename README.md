# Object Detection and Profile Picture Validation Project

This project provides a solution for object detection using the YOLO (You Only Look Once) deep learning model with OpenCV. It specifically includes functionality to detect objects in images and validate profile pictures based on criteria such as ensuring that there is only one human face in the image, and that the face occupies a specified percentage (e.g., 70% or 80%) of the image.

## Features

- **YOLO Object Detection:** Uses YOLO deep learning model for detecting objects in images.
- **Profile Picture Validation:** Ensures that images contain only one face and that the face occupies a defined percentage of the image area (e.g., 70% - 90%).
- **Customizable Detection Thresholds:** Allows configuration of confidence thresholds for object detection and non-maximum suppression (NMS) thresholds to filter out redundant bounding boxes.
- **Image Processing with OpenCV:** Uses the OpenCV library for image loading, preprocessing, and drawing bounding boxes.

## Requirements

- Java 8 or higher
- OpenCV 4.x or higher
- YOLO v3 or v4 configuration and weights files
- SLF4J (Simple Logging Facade for Java)
- A configuration file for model paths and OpenCV native library

## Libraries and Dependencies

To use this project, you will need the following libraries:

- **OpenCV**: For image processing and running the YOLO model.
- **SLF4J**: For logging.
- **YOLO**: The pre-trained YOLO model, including the configuration and weights files, is required for object detection.

You can install OpenCV and SLF4J using Maven or manually add the `.jar` files to your project.

## Installation and Setup

### Step 1: Install OpenCV

1. Download OpenCV 4.x from the [OpenCV website](https://opencv.org/releases/).
2. Add the OpenCV library to your Java project.
3. Ensure that you have the OpenCV native library path specified in your system's environment variable.

### Step 2: YOLO Weights and Configuration

The weights and configuration files are already provided in the `models` folder inside the `resources` directory.

### Step 3: Project Configuration

1. Clone or download the project from GitHub.
2. The `config.properties` file in the `resources` directory is already configured with the following settings:

```properties
opencv.library.path=/usr/lib/jni/libopencv_java454d.so

# Paths for YOLOv3 model
yolov3.weights.path=models/yolov3.weights
yolov3.configuration.path=models/yolov3.cfg
```

### Output

- The application will generate annotated images with bounding boxes drawn around detected objects.
- If a profile picture is validated, the application will log whether the image is valid or not based on the defined face percentage criteria.

## Usage

**Note:** Ensure that the OpenCV library is initialized by calling `OpenCVLoader.loadLibrary()` before using any of the detection functionalities. This can be done using a static block or any other suitable method.

### Minimal Usage Example

```java
import com.luni.od.YoloObjectDetector;
import com.luni.od.YoloObjectDetectorHelper;

import com.luni.od.OpenCVLoader;

public class MinimalUsageExample {
    static {
        OpenCVLoader.loadLibrary();
    }

    public static void main(String[] args) {
        YoloObjectDetectorHelper helper = new YoloObjectDetectorHelper();
        YoloObjectDetector detector = new YoloObjectDetector.Builder(helper.getModelConfiguration(),
                helper.getModelWeight()).build();
        
        // Validate if the image is a valid profile picture
        boolean isValid = detector.isValidDisplayPicture("/input/image.jpg");
        if (isValid) {
            System.out.println("The image is a valid profile picture.");
        } else {
            System.out.println("The image is not a valid profile picture.");
        }        
    }
}
```

### Running Object Detection and Validating Profile Pictures

### Code Samples for Using Main Classes

Here are some examples to help you get started with the different classes and how to use them effectively.

#### 1. Using the `YoloObjectDetector` Class

The `YoloObjectDetector` class is the main class used for detecting objects in images.

```java
import com.luni.od.YoloObjectDetector;
import com.luni.od.YoloObjectDetectorHelper;

import com.luni.od.OpenCVLoader;

public class YoloUsageExample {
    static {
        OpenCVLoader.loadLibrary();
    }

    public static void main(String[] args) {
        YoloObjectDetectorHelper helper = new YoloObjectDetectorHelper();
        YoloObjectDetector detector = new YoloObjectDetector.Builder(helper.getModelConfiguration(),
                helper.getModelWeight())
                .setConfidenceThreshold(0.5f)
                .setNmsThreshold(0.4f)
                .setMinFacePercentage(70.0f)
                .setMaxFacePercentage(90.0f)
                .build();


        // Validate if the image is a valid profile picture
        boolean isValid = detector.isValidDisplayPicture("/input/image.jpg");
        if (isValid) {
            System.out.println("The image is a valid profile picture.");
        } else {
            System.out.println("The image is not a valid profile picture.");
        }

        // Detect objects in an image and save the output
        detector.detectObjectsAndSave("/input/image.jpg", "/output/annotated_image.jpg", helper.getLabelMap());
    }
}
```

This code creates an instance of `YoloObjectDetector` using the builder pattern, which allows you to customize detection thresholds and other settings. Then, it processes an image and saves the output with bounding boxes drawn around detected objects.

#### 2. Using the `YoloObjectDetectorHelper` Class

The `YoloObjectDetectorHelper` class helps manage resources, such as model configuration and weights, required by the `YoloObjectDetector`.

```java
import com.luni.od.YoloObjectDetectorHelper;

import com.luni.od.OpenCVLoader;

public class HelperUsageExample {
    static {
        OpenCVLoader.loadLibrary();
    }

    public static void main(String[] args) {
        YoloObjectDetectorHelper helper = new YoloObjectDetectorHelper();
        String modelConfiguration = helper.getModelConfiguration();
        String modelWeights = helper.getModelWeight();

        System.out.println("Model Configuration Path: " + modelConfiguration);
        System.out.println("Model Weights Path: " + modelWeights);
    }
}
```

This class makes it easier to initialize the YOLO model by providing access to the model configuration and weight paths from the `config.properties` file.

#### 3. Why Use the `OpenCVLoader` Class

The `OpenCVLoader` class is responsible for loading the OpenCV native library, which is necessary for image processing and computer vision tasks. It ensures that the library is loaded only once to prevent multiple initializations.

```java
import com.luni.od.OpenCVLoader;

public class OpenCVLoaderUsageExample {
   static {
        OpenCVLoader.loadLibrary(); // Load OpenCV Library
    }

    public static void main(String[] args) {
        System.out.println("OpenCV library loaded successfully!");
        // You can now use OpenCV functionalities in your application.
    }
}
```

This code shows how to use the `OpenCVLoader` class to load the OpenCV library before using any OpenCV functionalities in your Java application.

### Configuration for CPU and GPU

You can configure the `YoloObjectDetector` to run inference on either the CPU or the GPU, depending on your system capabilities.

#### Example: Using CPU for Inference

By default, the `YoloObjectDetector` is set to use the CPU for inference. You do not need to change any configurations if you want to run the object detection on the CPU.

```java
YoloObjectDetector detector = new YoloObjectDetector.Builder(helper.getModelConfiguration(),
        helper.getModelWeight())
        .setPreferableBackend(Dnn.DNN_BACKEND_OPENCV)
        .setPreferableTarget(Dnn.DNN_TARGET_CPU) // Run on CPU
        .build();
```

#### Example: Using GPU for Inference

To use GPU acceleration (if available), you need to set the preferable backend and target accordingly.

```java
YoloObjectDetector detector = new YoloObjectDetector.Builder(helper.getModelConfiguration(),
        helper.getModelWeight())
        .setPreferableBackend(Dnn.DNN_BACKEND_CUDA) // Use CUDA backend for GPU
        .setPreferableTarget(Dnn.DNN_TARGET_CUDA) // Run on GPU
        .build();
```

This allows the YOLO model to use GPU acceleration, significantly speeding up inference times when a compatible GPU is available.

### Summary
- Use the `YoloObjectDetector` class to detect objects in images, customize thresholds, and save output images.
- Use the `YoloObjectDetectorHelper` to manage model configurations and resources.
- The `OpenCVLoader` ensures the native OpenCV library is loaded correctly, which is essential for the functionality of this project.
- You can configure the backend and target for either CPU or GPU inference to optimize performance.

## Logging

SLF4J is used for logging. By default, the logger is set to log INFO and ERROR levels to the console. You can configure SLF4J in your `logback.xml` or `log4j.properties` for custom logging levels or output formats.

## Known Issues and Limitations

- Ensure that OpenCV is correctly set up on your system. Failure to do so may cause runtime errors.
- The performance of YOLO on CPUs may be slower compared to GPUs. Consider using GPU acceleration if available.

## License

This project is licensed under the MIT License.

---

By following these instructions, you should be able to successfully set up and use the object detection and profile picture validation system based on YOLO and OpenCV.

Feel free to raise issues or contribute to the project via GitHub!

