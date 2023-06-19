import numpy as np
from elements.assets import transform_matrix
def interpolate_coordinates(ball):
    # Original position coordinate data
    x = [coord[0] if len(coord) > 0 else np.nan for coord in ball].copy()
    y = [coord[1] if len(coord) > 0 else np.nan for coord in ball].copy()
    # Create a time sequence as the interpolation basis
    t = np.arange(len(x))

    # Find the indices of missing coordinates
    missing_indices = [i for i, (xi, yi) in enumerate(zip(x, y)) if np.isnan(xi) or np.isnan(yi)]

    # Fill in the missing coordinates
    for i in missing_indices:
        x_known = np.array([xi for xi in x if not np.isnan(xi)])
        y_known = np.array([yi for yi in y if not np.isnan(yi)])
        x[i] = np.interp(t[i], t[~np.isnan(x)], x_known)
        y[i] = np.interp(t[i], t[~np.isnan(y)], y_known)

    # Fill in the missing coordinates in the original ball data
    for i, coord in enumerate(ball):
        if len(coord) == 0:
            ball[i] = [int(x[i]), int(y[i])]
    return ball

def velocity_2d(i_xy, rate):
    if len(i_xy) < 2:
        return 0, 0
    vx = 0
    vy = 0
    # weight = 1
    # rate = 0.95
    sum_rate = 0
    nf, xf, yf = i_xy[-1]
    for i in range(len(i_xy) - 1):
        n, x, y = i_xy[i]
        if(nf - n) != 0:
            weight = rate ** (nf - n)
            vx += (xf - x) / (nf - n) * weight
            vy += (yf - y) / (nf - n) * weight
            sum_rate += weight
            # weight *= rate
    if sum_rate != 0:
        vx /= sum_rate
        vy /= sum_rate
    return vx, vy

def search_good_M(M, c_xy, hw, gt_hw):
    (x_center, y_center) = c_xy
    (h, w) = hw
    (gt_h, gt_w) = gt_hw
    n_range = 12
    n_vol = 10
    good_M = []
    temp_M = []
    temp_v_xy = []
    good_M.append(0)
    xy = transform_matrix(M[0], (x_center, y_center), (h, w), (gt_h, gt_w))
    temp_v_xy.append([0, xy[0], xy[1]])
    for i in range(1, len(M)):
        temp_M.append(M[i])
        if(i % n_range == 0):
            last = transform_matrix(M[good_M[-1]], (x_center, y_center), (h, w), (gt_h, gt_w))
            (lx, ly) = last
            min_loss = 1000
            vx, vy = velocity_2d(temp_v_xy, 0.95)
            Mn = -1
            for j, Mi in enumerate(temp_M):
                center = transform_matrix(Mi, (w/2, h/2), (h, w), (gt_h, gt_w))
                (x, y) = center
                pre_x = lx + vx * (i - n_range + j + 1 - good_M[-1])
                pre_y = ly + vy * (i - n_range + j + 1 - good_M[-1])
                loss = (pre_x - x)**2 + (pre_y - y)**2
                if(loss < min_loss):
                    min_loss = loss
                    Mn = j
            index = i - n_range + Mn + 1
            good_M.append(index)
            now = transform_matrix(M[good_M[-1]], (x_center, y_center), (h, w), (gt_h, gt_w))
            temp_v_xy.append([index, now[0], now[1]])
            temp_M = []
            if(len(temp_v_xy) > n_vol):
                temp_v_xy.pop(0)
    good_M.append(len(M) - 1)
    return good_M

def interpolate_M(M, good_M):
    for i in range(len(good_M) - 1):
        b = good_M[i]
        e = good_M[i+1]
        if (e - b) > 1:
            for j in range(b+1, e):
                Mb = M[b]
                Me = M[e]
                MM = []
                t = []
                for r in range(3):
                    for c in range(3):
                        t.append(Mb[r][c] + (Me[r][c] - Mb[r][c])/(e-b)*(j-b))
                    MM.append(t)
                    t = []
                M[j] = MM
    # if good_M[-1] != (len(M) - 1):
    #     for i in range(good_M[-1] + 1, len(M)):
    #         M[i] =  M[good_M[-1]]
    return M

def optimize_M(M, c_xy, hw, gt_hw):
    good_M = search_good_M(M, c_xy, hw, gt_hw)
    M = interpolate_M(M, good_M)
    return M


# M = [[[ 0.00045432,   0.0012674,    -0.19591],
#  [ 2.5538e-05,   0.0030347,    -0.77801],
#  [-4.4897e-07,  2.1818e-05,   0.0027305]],
#  [[ 0.00064224,   0.0012713,    -0.23984],
#  [ 0.00039053,   0.0040699,     -1.0971],
#  [ 4.7695e-06,  2.1567e-05,  0.00052298]],
#  [[ 0.00064224,   0.0012713,    -0.23984],
#  [ 0.00039053,   0.0040699,     -1.0971],
#  [ 4.7695e-06,  2.1567e-05,  0.00052298]],
#  [[ 0.00064224,   0.0012713,    -0.23984],
#  [ 0.00039053,   0.0040699,     -1.0971],
#  [ 4.7695e-06,  2.1567e-05,  0.00052298]],
#  [[ 0.00064224,   0.0012713,    -0.23984],
#  [ 0.00039053,   0.0040699,     -1.0971],
#  [ 4.7695e-06,  2.1567e-05,  0.00052298]],
#  [[ 0.00064224,   0.0012713,    -0.23984],
#  [ 0.00039053,   0.0040699,     -1.0971],
#  [ 4.7695e-06,  2.1567e-05,  0.00052298]],
#  [[ 0.00064224,   0.0012713,    -0.23984],
#  [ 0.00039053,   0.0040699,     -1.0971],
#  [ 4.7695e-06,  2.1567e-05,  0.00052298]],
#  [[ 0.00064224,   0.0012713,    -0.23984],
#  [ 0.00039053,   0.0040699,     -1.0971],
#  [ 4.7695e-06,  2.1567e-05,  0.00052298]],
#  [[ 0.00064224,   0.0012713,    -0.23984],
#  [ 0.00039053,   0.0040699,     -1.0971],
#  [ 4.7695e-06,  2.1567e-05,  0.00052298]],
#  [[ 0.00064224,   0.0012713,    -0.23984],
#  [ 0.00039053,   0.0040699,     -1.0971],
#  [ 4.7695e-06,  2.1567e-05,  0.00052298]],
#  [[ 0.00064224,   0.0012713,    -0.23984],
#  [ 0.00039053,   0.0040699,     -1.0971],
#  [ 4.7695e-06,  2.1567e-05,  0.00052298]],
#  [[ 0.00064224,   0.0012713,    -0.23984],
#  [ 0.00039053,   0.0040699,     -1.0971],
#  [ 4.7695e-06,  2.1567e-05,  0.00052298]]]
# print(M)
# print('---------------------------------------------------------------------------------------------------------')
# M = optimize_M(M, (640, 360), (720, 1280), (296, 460))
# print(M)