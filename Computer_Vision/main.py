"""
***********************************************
MACHINE LEARNING - Computer Vision
***********************************************
Author: Adam Gużewski



To run the program you should type in terminal: python main.py
You also need chromedriver to let the application control Chrome

My program is going to detect if the user has eyes open. If not it will pause the commercials.
"""

# Importing libraries
import time
import cv2 as cv
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# URL for Commercials
url = 'https://www.youtube.com/watch?v=HSRieuzms24&ab_channel=7Trendz'
chromedriver_path = 'chromedriver/chromedriver.exe'
button_text = "ZGADZAM SIĘ"

cap = cv.VideoCapture(0)


def rescale_frame(frame, percent=75):
    """

    :param frame:
    :param percent:
    :return:
    """
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)


face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")

browser_service = Service(chromedriver_path)
browser = webdriver.Chrome(service=browser_service)
# browser.maximize_window()
browser.get(url)
WebDriverWait(browser, 20).until(EC.presence_of_element_located((By.LINK_TEXT, button_text))).click()
# time.sleep(2)

# For new window in chrome I have to accept privacy policy

# agreeBtn = browser.find_element(By.LINK_TEXT, button_text)
# webdriver.ActionChains(browser).move_to_element(agreeBtn).perform()
# time.sleep(1)
# webdriver.ActionChains(browser).click(agreeBtn).perform()
print(browser.title)

video = browser.find_element(By.ID, 'movie_player')
WebDriverWait(video, 5).until(EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Wstrzymaj (k)']"))).click()
time.sleep(1)
WebDriverWait(video, 5).until(EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Odtwórz (k)']"))).click()

pause_btn = browser.find_element(By.XPATH, "//button[@aria-label='Wstrzymaj (k)']")
print(pause_btn)
# pause_btn = browser.find_element(By.XPATH, "//button[@aria-label='Play']")
# print(pause_btn)
# video.send_keys(Keys.SPACE)


while (True):
    ret, frame = cap.read()
    # cv.imshow('You have to watch it!', frame)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        roi_gray = gray[y:y + w, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 15)
        if len(eyes) == 0:
            print("Eyes Closed!")

        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)

    rescaled_frame = rescale_frame(frame, percent=150)
    cv.imshow('frame', rescaled_frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
