import cv2
import numpy as np

img = cv2.imread("img/taekwonv1.jpg")

# ROI선택하기
x, y, w, h = cv2.selectROI('image', img, False)
if w and h:
    roi = img[y:y + h, x:x + w]
    cv2.imshow("image", roi)

# roi gary 이미지 생성
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# 이미지 노멀라이즈
norm_roi = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)
norm_gary_roi = cv2.normalize(gray_roi, None, 0, 255, cv2.NORM_MINMAX)
# cv2.imshow("selectROI", norm_roi)


# 캐니 엣지 생성
edge = cv2.Canny(norm_gary_roi, 10, 100)
#cv2.imshow("edge", edge)


# 스레스 홀드를 이용하여 이진화 (inv: 검정배경, 하얀물체)
ret, th_inv = cv2.threshold(norm_gary_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
ret, th = cv2.threshold(norm_gary_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
merged = np.hstack((th_inv, th))
# cv2.imshow("th_inv_ori", merged)


# 스레스홀드 inv와 edge bitwise_or를 통해 테두리 만들기
th_inv_edge = cv2.bitwise_or(edge, th_inv)
# cv2.imshow("th_edge", th_inv_edge)


# 모폴로지를 통해 테두리 두껍게 잡기
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
morph = cv2.morphologyEx(th_inv_edge, cv2.MORPH_DILATE, k, None, None, 1)
morph = cv2.morphologyEx(morph, cv2.MORPH_ERODE, k, None, None, 1)
morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, k, None, None, 1)
# cv2.imshow("morph", morph)


# 인터넷 참조(46~63)====================================================================
# floodfill을 이용해 물체 마스크 만들기
flood = morph.copy()
h, w = morph.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

# 검정 배경을 흰색으로 바꾸기 (흰색 배경, 검정 부분)
cv2.floodFill(flood, mask, (0, 0), 255)
# cv2.imshow("flood", flood)

# bitwise_not을 이용하여 색 반전 (검정 배경, 흰색 부분)
flood_inv = cv2.bitwise_not(flood)
# cv2.imshow("flood_inv", flood_inv)

# bitwise_or을 이용하여 물체 마스크 만들기
sth_mask = cv2.bitwise_or(flood_inv, morph)
# cv2.imshow("sth_mask", sth_mask)
# ===================================================================================

# 2번째 물체 마스크 만들기
# 적응형 스레스 홀드
adapt_th_inv = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5)
adapt_th = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)
merged = np.hstack((adapt_th, adapt_th_inv))
# cv2.imshow("adapt_th, inv", merged)

# 모폴로지를 통해 테두리 두껍게 잡기
adapt_morph = cv2.morphologyEx(adapt_th_inv, cv2.MORPH_CLOSE, k, None, None, 2)
# cv2.imshow("adapt_morph", adapt_morph)

# floodfill이용 배경색 흰색으로 바꾸기
flood2 = adapt_morph.copy()
mask = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(flood2, mask, (0, 0), 255)
# cv2.imshow("flood2", flood2)

# bitwise_not을 이용하여 색 반전 (검정 배경, 흰색 부분)
flood2_inv = cv2.bitwise_not(flood2)
# cv2.imshow("flood2_inv", flood2_inv)

# bitwise_or을 이용하여 물체 마스크 만들기
sth_mask2 = cv2.bitwise_or(flood2_inv, adapt_morph)
#cv2.imshow("sth_mask2", sth_mask2)

# 물체 마스크1과 물체 마스크2 bitwise_or
mask_re = cv2.bitwise_or(sth_mask, sth_mask2)
#cv2.imshow("mask", mask_re)

# 모폴로지 이용 -> 마스크 빈 부분 채우기
k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask_re = cv2.morphologyEx(mask_re, cv2.MORPH_CLOSE, k2, None, None, 1)
# cv2.imshow("maskre", mask_re)

# 메디안블러를 통한 잡티 제거
blur = cv2.medianBlur(mask_re, 9)
# cv2.imshow("blur", blur)
mask = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
cv2.imshow("mask", mask)

# 이미지 분리
sth = cv2.bitwise_and(norm_roi, mask)
cv2.imshow("something", sth)

# 이미지 hsv값 제한
sth_hsv = cv2.cvtColor(sth, cv2.COLOR_BGR2HSV)
for i in range(h):
    for j in range(w):
        if sth_hsv[i, j, 2] == 0:
            continue
        hue = sth_hsv[i, j, 0] / 10
        sth_hsv[i, j, 0] = hue * 10 + 5
        s = sth_hsv[i, j, 1] % 5
        sth_hsv[i, j, 1] -= s
        v = sth_hsv[i, j, 2] % 20
        sth_hsv[i, j, 2] -= v

sth = cv2.cvtColor(sth_hsv, cv2.COLOR_HSV2BGR)
sth_tmp = sth.copy()
mask_tmp = mask.copy()
cv2.imshow("something", sth)


# 제대로 마스크 처리 안된부분 처리하기 위한 마우스 이벤트
def onMouse(event, x, y, flags, param):
    global sth_tmp
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(sth_tmp, (x, y), 3, (0, 0, 0), -1)
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            cv2.circle(sth_tmp, (x, y), 7, (0, 0, 0), -1)

    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(sth_tmp, (x, y), 3, (0, 0, 0), -1)
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            cv2.circle(sth_tmp, (x, y), 7, (0, 0, 0), -1)

    cv2.imshow("something", sth_tmp)

# 마우스 이벤트 처리를 위한 반복문
while True:
    cv2.setMouseCallback("something", onMouse)
    key = cv2.waitKey(0)
    if key == ord('r'):
        sth_tmp = sth.copy()
        mask_tmp = mask.copy()
        cv2.imshow("something", sth_tmp)

    elif key == ord('n'):
        sth = sth_tmp.copy()
        mask = mask_tmp
        cv2.imshow("something", sth)
        break
    elif key == 27:
        cv2.destroyAllWindows()

# 픽셀값 지정
px48 = (48, 48)
px64 = (64, 64)
px128 = (128, 128)

# 128bit 이미지 생성
sth128 = cv2.resize(sth, px128)

sth_gray = cv2.cvtColor(sth128, cv2.COLOR_BGR2GRAY)
h2, w2, _ = np.shape(sth128)
print(h2, w2)
# 배경 투명화
sth_alpha = cv2.cvtColor(sth128, cv2.COLOR_BGR2BGRA)
for i in range(h2):
    for j in range(w2):
        if sth_gray[i, j] == 0:
            sth_alpha[i, j, 3] = 0

# 테두리 처리
for i in range(h2):
    for j in range(w2):
        if sth_alpha[i, j, 3] == 0:
            continue
        if sth_alpha[i, j, 3] != 0 and (sth_alpha[i, j-1, 3] == 0 or sth_alpha[i-1, j, 3] == 0):
            sth_alpha[i, j] = (0, 0, 0, 255)
        if sth_alpha[i, j, 3] == 255 and (sth_alpha[i, j+1, 3] == 0 or sth_alpha[i+1, j, 3] == 0):
            sth_alpha[i, j] = (0, 0, 0, 255)

sth128_alpha = cv2.resize(sth_alpha, (w, h), interpolation=cv2.INTER_AREA)


# 64bit이미지 생성
sth64 = cv2.resize(sth, px128)
sth64 = cv2.resize(sth64, (w, h), interpolation=cv2.INTER_AREA)
sth64 = cv2.resize(sth64, px64)

sth_gray = cv2.cvtColor(sth64, cv2.COLOR_BGR2GRAY)
h2, w2, _ = np.shape(sth64)

# 배경 투명화
sth_alpha = cv2.cvtColor(sth64, cv2.COLOR_BGR2BGRA)
for i in range(h2):
    for j in range(w2):
        if sth_gray[i, j] == 0:
            sth_alpha[i, j, 3] = 0

# 테두리 처리
for i in range(h2):
    for j in range(w2):
        if sth_alpha[i, j, 3] == 0:
            continue
        if sth_alpha[i, j, 3] != 0 and (sth_alpha[i, j-1, 3] == 0 or sth_alpha[i-1, j, 3] == 0):
            sth_alpha[i, j] = (0, 0, 0, 255)
        if sth_alpha[i, j, 3] == 255 and (sth_alpha[i, j+1, 3] == 0 or sth_alpha[i+1, j, 3] == 0):
            sth_alpha[i, j] = (0, 0, 0, 255)

sth64_alpha = cv2.resize(sth_alpha, (w, h), interpolation=cv2.INTER_AREA)


# 48bit 이미지 생성
sth48 = cv2.resize(sth, px128)
sth48 = cv2.resize(sth48, (w, h), interpolation=cv2.INTER_AREA)
sth48 = cv2.resize(sth48, px48)

sth_gray = cv2.cvtColor(sth48, cv2.COLOR_BGR2GRAY)
h2, w2, _ = np.shape(sth48)

# 배경 투명화
sth_alpha = cv2.cvtColor(sth48, cv2.COLOR_BGR2BGRA)
for i in range(h2):
    for j in range(w2):
        if sth_gray[i, j] == 0:
            sth_alpha[i, j, 3] = 0

# 테두리 처리
for i in range(h2):
    for j in range(w2):
        if sth_alpha[i, j, 3] == 0:
            continue
        if sth_alpha[i, j, 3] != 0 and (sth_alpha[i, j-1, 3] == 0 or sth_alpha[i-1, j, 3] == 0):
            sth_alpha[i, j] = (0, 0, 0, 255)
        if sth_alpha[i, j, 3] == 255 and (sth_alpha[i, j+1, 3] == 0 or sth_alpha[i+1, j, 3] == 0):
            sth_alpha[i, j] = (0, 0, 0, 255)

sth48_alpha = cv2.resize(sth_alpha, (w, h), interpolation=cv2.INTER_AREA)

merged = np.hstack((sth128_alpha, sth64_alpha, sth48_alpha))
cv2.imshow("result", merged)

cv2.imwrite("img/result128.png", sth128_alpha)
cv2.imwrite("img/result64.png", sth64_alpha)
cv2.imwrite("img/result48.png", sth48_alpha)

cv2.waitKey(0)
cv2.destroyAllWindows()

