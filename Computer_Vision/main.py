"""
***********************************************
MACHINE LEARNING - Computer Vision
***********************************************
Author: Adam Gużewski

To run the program you should type in terminal: python main.py
You also need chromedriver to let the application control Chrome

My program is going to detect if the user has eyes open. If not it will pause the commercials.
If The User opens the eyes, the application will resume the video with ads.
"""

# Importing libraries
import time
import cv2 as cv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# URL for Commercials
url = 'https://www.youtube.com/watch?v=LeQ1EL0Cf0A'
# Path to chromedriver
chromedriver_path = 'chromedriver/chromedriver.exe'
# Button to agree to terms of use
button_text = "ZGADZAM SIĘ"

cap = cv.VideoCapture(0)


def rescale_frame(frame, percent=75):
    """
    Function will resize input video to the size of given percentage
    :param frame: input video
    :param percent: scaling percentage
    :return: The function will return rescaled frame
    """
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)

# Loading Haar Cascade Classifiers
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")

# Choosing Chrome
browser_service = Service(chromedriver_path)
# Creating an instance of Chrome
browser = webdriver.Chrome(service=browser_service)
browser.maximize_window()
# Opening Ads
browser.get(url)
# For new window in chrome I have to accept privacy policy
WebDriverWait(browser, 20).until(EC.presence_of_element_located((By.LINK_TEXT, button_text))).click()

print(browser.title)

video = browser.find_element(By.ID, 'movie_player')
# I need to pause the app to load the window and pause button
time.sleep(1)
pause_btn = browser.find_element(By.XPATH, "//button[@aria-label='Wstrzymaj (k)']")

status = 'play'
while True:
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 176, 34), 2)
        roi_gray = gray[y:y + w, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 15)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (96, 215, 30), 2)

        if len(eyes) > 0:
            if status == 'play':
                continue
            elif status == 'pause':
                status = 'play'
                WebDriverWait(video, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Odtwórz (k)']"))).click()
                print('Playing advertisements')
                continue
        if len(eyes) == 0:
            if status == 'play':
                status = 'pause'
                pause_btn.click()
                print('Ads stopped!')
                continue
            elif status == 'pause':
                continue

    rescaled_frame = rescale_frame(frame, percent=150)
    cv.imshow('You have to watch it!', rescaled_frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
browser.quit()
