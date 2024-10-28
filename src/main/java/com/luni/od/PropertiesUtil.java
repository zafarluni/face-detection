package com.luni.od;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class PropertiesUtil {
    private static PropertiesUtil instance;
    private Properties properties;
    private final String propertiesFileName = "config.properties"; // Fixed properties file name

    // Private constructor to prevent instantiation
    private PropertiesUtil() {
        properties = new Properties();
        loadProperties();
    }

    // Static method to get the singleton instance
    public static PropertiesUtil get() {
        if (instance == null) {
            synchronized (PropertiesUtil.class) { // Thread-safe initialization
                if (instance == null) {
                    instance = new PropertiesUtil();
                }
            }
        }
        return instance;
    }

    // Load properties from the specified file
    private void loadProperties() {
        try (InputStream input = getClass().getClassLoader().getResourceAsStream(propertiesFileName)) {
            if (input == null) {
                System.out.println("Sorry, unable to find " + propertiesFileName);
                return;
            }
            properties.load(input);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Method to get a property value by key
    public String getProperty(String key) {
        return properties.getProperty(key);
    }

    // Method to get a property value with a default value
    public String getProperty(String key, String defaultValue) {
        return properties.getProperty(key, defaultValue);
    }
}
