/**
 * OpenCVLoader is a utility class responsible for loading the OpenCV native 
 * library required for image processing and computer vision tasks. This class 
 * ensures that the OpenCV library is loaded only once to prevent multiple 
 * initializations, which can lead to errors and inefficiencies.
 * 
 * The class provides a method to load the OpenCV library by retrieving the 
 * library path from a properties file. Once the library is successfully loaded, 
 * subsequent calls to load the library are ignored, ensuring a singleton 
 * loading behavior.
 * 
 * This class is essential for applications that utilize OpenCV functionalities, 
 * allowing developers to focus on implementing features without worrying about 
 * library loading issues.
 * 
 * Author: Zafar Hussain Luni
 */

package com.luni.od;

public class OpenCVLoader {
    private static boolean isLoaded = false;

    private OpenCVLoader() {
    }

    public static void loadLibrary() {
        if (!isLoaded) {
            System.load(PropertiesUtil.get().getProperty("opencv.library.path"));
            isLoaded = true;
        }
    }
}
