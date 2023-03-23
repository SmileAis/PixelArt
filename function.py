import cv2
import numpy as np

# 제대로 마스크 처리 안된부분 처리하기 위한 마우스 이벤트
def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(param, (x, y), 3, (0, 0, 0), -1)
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            cv2.circle(param, (x, y), 7, (0, 0, 0), -1)

    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(param, (x, y), 3, (0, 0, 0), -1)
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            cv2.circle(param, (x, y), 7, (0, 0, 0), -1)

    cv2.imshow("something", param)
    
    
# 이미지 픽셀화
def img_to_pixel(size, sth, w, h):
    pixel = (size, size);
    pix_img = cv2.resize(sth, (128, 128))

    if size != 128:
        pix_img = cv2.resize(pix_img, (w, h), interpolation=cv2.INTER_AREA)
        pix_img = cv2.resize(pix_img, pixel)

    pix_gray = cv2.cvtColor(pix_img, cv2.COLOR_BGR2GRAY)
    h2, w2, _ = np.shape(pix_img)

    # 배경 투명화
    pix_alpha = cv2.cvtColor(pix_img, cv2.COLOR_BGR2BGRA)
    for i in range(h2):
        for j in range(w2):
            if pix_gray[i, j] == 0:
                pix_alpha[i, j, 3] = 0

    # 테두리 처리
    for i in range(h2):
        for j in range(w2):
            if pix_alpha[i, j, 3] == 0:
                continue
            if pix_alpha[i, j, 3] != 0 and (pix_alpha[i, j - 1, 3] == 0 or pix_alpha[i - 1, j, 3] == 0):
                pix_alpha[i, j] = (0, 0, 0, 255)
            if pix_alpha[i, j, 3] == 255 and (pix_alpha[i, j + 1, 3] == 0 or pix_alpha[i + 1, j, 3] == 0):
                pix_alpha[i, j] = (0, 0, 0, 255)

    res = cv2.resize(pix_alpha, (w, h), interpolation=cv2.INTER_AREA)

    return res