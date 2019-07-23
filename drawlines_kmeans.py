import cv2
from directkeys import PressKey, ReleaseKey
from grabscreen import grab_screen
import numpy as np
import pyautogui
from sklearn.cluster import KMeans
import threading
import time

uhd_x = 640
uhd_y = 312
BOX = (uhd_x, uhd_y, uhd_x + 640, uhd_y + 480)

hood_y = 410
horizon_y = 175
side_y = 50
keytime = 0.1

STRAIGHT = 0x11
RIGHT = 0x20
LEFT = 0x1E

VERTICES = np.array([[4, horizon_y + side_y],
                     [220, horizon_y], [580, horizon_y],
                     [800, horizon_y + side_y],
                     [800, hood_y], [4, hood_y]])

for i in range(3, 0, -1):
    time.sleep(.4)
    print(i)


def t_key(key_a, key_b, key_c):
    PressKey(key_a)
    ReleaseKey(key_b)
    ReleaseKey(key_c)
    time.sleep(keytime)


def straight():
    print("pressing up")
    thread_straight = threading.Thread(target=t_key,
                                       args=(STRAIGHT, RIGHT, LEFT))
    thread_straight.start()


def right():
    print("pressing r")
    thread_right = threading.Thread(target=t_key,
                                    args=(RIGHT, STRAIGHT, LEFT))
    thread_right.start()


def left():
    print("pressing l")
    thread_left = threading.Thread(target=t_key,
                                   args=(LEFT, STRAIGHT, RIGHT))
    thread_left.start()


def slope(line):
    try:
        y = line[1] - line[3]
        x = line[0] - line[2]
        slope = np.divide(y, x)
    except ZeroDivisionError:
        slope = 100000
    finally:
        return slope


def drive(m=None):
    sign = np.sum(np.sign(m))
    if sign == -2:
        right()
    elif sign == 2:
        left()
    else:
        straight()


def draw_lines(img, lines):
    try:
        m = []
        for coords in lines:
            m.append(slope(coords))
            coords = np.array(coords, dtype='uint32')
            cv2.line(img,
                     (coords[0], coords[1]),
                     (coords[2], coords[3]),
                     [255, 255, 255], 10)
    except TypeError as e:
        print('draw lines error: {}'.format(e))
    else:
        pass
        drive(m)


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_img(original_img):

    processed_img = cv2.Canny(original_img,
                              threshold1=100, threshold2=300)

    processed_img = roi(processed_img, [VERTICES])

    # processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
    lines = cv2.HoughLinesP(processed_img, 1,
                            np.pi / 180, 180, np.array([]), 120, 20)

    # draw_lines(processed_img, nlines)

    try:
        nlines = np.array([l[0] for l in lines])
        kmeans = KMeans(n_clusters=2, random_state=0).fit(nlines)
        draw_lines(processed_img, kmeans.cluster_centers_)
    except (ValueError, TypeError) as e:
        print('Kmeans error: {}'.format(e))

    return processed_img


def main():
    while True:
        ti = time.time()
        screen = grab_screen(region=BOX)
        cv2.imshow('window', cv2.cvtColor(screen,
                                          cv2.COLOR_BGR2RGB))

        new_screen = process_img(screen)
        cv2.imshow('window2', new_screen)

        print('{:.2f} FPS'.format(1 / (time.time() - ti)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
