import sys
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path('01_camera_calib').resolve()))
import calib_lib as cl
import selftest_synthetic as ss

calib = cl.load_calib('01_camera_calib/out/calib.json')
K, dist = calib['camera_matrix'], calib['dist_coeffs']
print('K', K.tolist())
print('dist', dist.ravel().tolist())
ss.SQ = 19.0
tex = ss.board_texture()
print('tex', tex.shape)
rng = np.random.default_rng(0)
# board object center
pat = (8,5)
objp = cl.object_points(pat, 19.0)
center = objp.mean(axis=0)
print('center', center, 'obj range', objp.min(0), objp.max(0))
for i, rv in enumerate([np.zeros(3), np.array([0.15,-0.2,0.05]), np.array([-0.2,0.25,-0.08])]):
    R = cv2.Rodrigues(rv)[0]
    for Z in [450.0, 550.0]:
        t = np.array([0.0,0.0,Z]) - R@center
        img = ss.render_view(K, dist, rv, t, tex, (1280,720), rng)
        ok, corners = cl.find_corners(img, pat)
        print(f'pose {i} Z={Z} corners={ok}', end=' ')
        if ok:
            rvec2,tvec2,rms = cl.solve_pose(corners, pat, 19.0, K, dist)
            print(f'rms={rms:.3f} t={np.round(tvec2.ravel(),1)} rv={np.round(rvec2.ravel(),3)}')
        else:
            print()
