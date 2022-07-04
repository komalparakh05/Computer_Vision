import streamlit as st
import numpy as np
import cv2 as cv

# Object lower and higher hue (for HSV mask purpose)
profils = {
    "Orange": {
        "lower": 15,
        "higher": 31
    },
    "Apple": {
        "lower": 0,
        "higher": 14
    },
    "Pear": {
        "lower": 32,
        "higher": 80
    }
}

# Load image.
uploaded_file = st.sidebar.file_uploader("Upload an image:")

if uploaded_file is not None:

    # convert string data to numpy array
    npimg = np.fromstring(uploaded_file.getvalue(), np.uint8)
    # convert numpy array to image
    frame = cv.imdecode(npimg, cv.IMREAD_COLOR)

    # Switch to HSV for simplier color handling.
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # lower_h, upper_h = st.slider("Select a hue range:", 0, 360, (0,360))

    # Loop on all objects we want to detect.
    for object_name in profils.keys():
        # Set lower and high hue for mask filtering.
        lower_h = profils[object_name]["lower"]
        upper_h = profils[object_name]["higher"]

        # Create the mask.
        mask = cv.inRange(hsv, (lower_h, 100, 100), (upper_h, 255, 255))

        # st.image(mask)

        # Find the object based on the mask.
        contours, hierarchy = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # If we found something.
        if contours:
            # Get the biggest contour
            contour = sorted(contours, key=cv.contourArea, reverse=True)[0]

            # To see all detected boxes.
            # for contour in contours:

            # Create rectangle from the biggest contour
            rect = cv.boundingRect(contour)
            x, y, w, h = rect

            # Draw rectangle on original image
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

            # Add name of the object
            cv.putText(frame, object_name, (x, y),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the result.
    # Convert to RGB for matplotlib proper color rendering.
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    st.image(frame)
