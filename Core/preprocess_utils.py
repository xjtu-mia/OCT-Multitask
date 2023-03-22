import cv2
import numpy as np
from scipy import signal


def intp(fp):
    fp_ = fp.copy().tolist()
    xp_ = np.arange(len(fp_))
    original_length = len(fp)
    index = []
    start = 0
    end = len(fp_) - 1
    flag = 0
    t = []
    for i in range(len(fp)):
        if fp[i] != 0 and flag == 1:

            if len(t) > 0:
                if t[0] == 0:
                    start = t[-1]
                if t[0] > 0:
                    index.append(t)
            t = []
            flag = 0

        elif fp[i] == 0:
            t.append(i)
            fp_.remove(0)
            flag = 1
    if len(t) > 0:
        end = t[0]

    s = fp_[0]
    for i in range(start + 1):
        # print('start', start)
        if start > 0:
            fp_.insert(i, s)  # 起始点未知则默认插入最近的一个有值点的值
    e = fp_[-1]
    for i in range(end, len(fp), 1):
        # print('end', end)
        if end <= len(fp) - 1:
            fp_.insert(i, e)  # 结束点未知则默认插入最近的一个有值点的值
    xp_ = np.arange(len(fp_))

    for group in index:
        v = np.linspace(group[0] - 1, group[0], group[-1] - group[0] + 3)
        x = np.array(v[1:-1])
        # print('x', x)
        # print('xp_', xp_)
        # print('fp_', fp_)
        int = np.interp(x, xp_, fp_)
        # print('group', group)
        # print('int', int)
        # print(int.shape)
        for i, y in enumerate(int):
            fp_.insert(group[0] + i, y)
        xp_ = np.arange(len(fp_))
    if len(fp_) < original_length:
        for l in range(original_length - len(fp_)):
            fp_.insert(len(fp_) + l, fp_[-1])
    return np.round(np.array(fp_))


def medfiter(X, win_len):
    Y = np.zeros_like(X)

    padwidth = int((win_len - 1) / 2)

    X = np.pad(X, ((0, 0), (padwidth, padwidth)))

    for i in range(padwidth, X.shape[1] - padwidth):
        Y[:, i - padwidth] = np.median(X[:, i - padwidth:i + padwidth + 1], axis=1)

    return Y


def np_move_avg(a, n, mode="valid"):
    shape = a.shape[0]
    if shape <= n:
        n = shape
    new_shape = shape + n - 1
    print(shape, new_shape, n)
    a_new = np.zeros(new_shape, dtype=a.dtype)

    a_new[0:shape] = a
    a_new[shape:new_shape] = a[-1 - (n - 1):-1]
    # print(a)
    # print(a_new)
    a_new = (np.convolve(a_new, np.ones((n,)) / n, mode=mode))
    return a_new


def roi(img):
    h, w = img.shape
    max = np.max(img)
    # cv2.imshow('i', img)
    ret, thr = cv2.threshold(img, 253, 254, cv2.THRESH_BINARY)
    # cv2.imshow('thr', thr)
    NpKernel = np.uint8(np.ones((7, 7)))
    thr = cv2.erode(thr, NpKernel)
    # 显示腐蚀后的图
    # 膨胀图像
    thr = cv2.dilate(thr, NpKernel)
    up = []
    down = []
    for i in range(w):
        y1 = 0
        y2 = 0
        y = thr[:, i]
        for j in range(h - 1):
            if y[j] == 255 and y[j + 1] == 0:
                y1 = j
                break
            else:
                y1 = 0
        for j in range(h - 1, 0, -1):
            if y[j] == 255 and y[j - 1] == 0:
                y2 = j
                break
            else:
                y2 = h - 1
        up.append(y1)
        down.append(y2)
    print(up, down)
    img = img[np.max(up): np.min(down), 0:w]
    return img


def flat_oct(img, y_index, dis=80, height=352):
    h, w = img.shape
    flat_img = np.zeros((h + dis, w))
    new_img = np.zeros_like(flat_img)
    new_y_index = np.zeros_like(y_index)
    baseline = h - dis
    for i in range(w):
        new_img[0:h, i] = img[:, i]
        col = img[:, i]
        bottom = np.max(y_index)
        dis1 = int(bottom - y_index[i])
        dis2 = int(h - dis - bottom)
        for j in range(dis1, h - dis2, 1):
            # print(j-dis1, i,j+dis2, i)
            flat_img[j + dis2, i] = new_img[j - dis1, i]
    return flat_img[h - height: h, 0: w], baseline


def unflat_oct(img, y_index1, baseline, size=(512, 512)):
    h, w = img.shape
    img_ = np.zeros(size, dtype=np.uint8)
    img_[(size[0] - h):, :] = img
    new_img = np.zeros_like(img_)
    # new_img = np.zeros(size)

    for i in range(w):
        dis = baseline - y_index1[i]
        # print(dis,h,w)
        if dis > 0:
            new_img[:size[0] - dis, i] = img_[dis:, i]
        else:
            new_img[abs(dis):, i] = img_[:size[0] - abs(dis), i]
    return new_img


def fill_boundary(img, filled_value=0, value='same'):
    h, w = img.shape
    boundary = np.zeros((w))
    for j in range(w):
        flag = 0
        for i in range(h // 3):
            if img[i, j] != filled_value:
                flag += 1
                if flag == 3:
                    for k in range(i):
                        if value == 'same':
                            img[k, j] = img[i, j]
                        else:
                            img[k, j] = value
                        boundary[j] = i + 10  # 视野边界坐标
                    break
        flag = 0
        for i in range(h - 1, h // 3 * 2, -1):
            if img[i, j] != filled_value:
                flag += 1
                if flag == 3:
                    for k in range(h - 1, i, -1):
                        if value == 'same':
                            img[k, j] = img[i, j]
                        else:
                            img[k, j] = value
                    break
    return img, boundary


def intensity_normalization(img):
    def rescale_linear(array, new_min, new_max):
        """Rescale an arrary linearly."""
        minimum, maximum = np.min(array), np.max(array)
        m = (new_max - new_min) / (maximum - minimum)
        b = new_min - m * minimum
        return m * array + b

    lm = 0
    for i in range(img.shape[1]):
        col = img[:, i]
        md = signal.medfilt(col, 15)
        lm = max(np.max(md), lm)
    lm = lm * 1.05
    print(lm)
    img[np.where(img > lm)] = lm
    # img[:, i] = rescale_linear(col, 0, 1)
    img = rescale_linear(img, 0, 1) * 255
    return img.astype(np.uint8)


def ConvexHull(img):
    img = img.astype(np.uint8)
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 图片轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    # 寻找凸包并绘制凸包（轮廓）
    hull = cv2.convexHull(cnt)
    # print(hull)
    length = len(hull)
    im = np.zeros_like(img)
    for i in range(len(hull)):
        cv2.line(im, tuple(hull[i][0]), tuple(hull[(i + 1) % length][0]), 255, 1)

    # cv2.imshow('1111', im)
    y_index = np.zeros(im.shape[1])
    for i in range(im.shape[1]):
        if i > 0 and i < im.shape[1] - 1:
            y_index[i] = np.argmax(im[1:-1, i], axis=0)
        else:
            for n in range(im.shape[0] - 1, 0, -1):
                if im[n, i] > 0:
                    y_index[i] = n
                    break
    return y_index