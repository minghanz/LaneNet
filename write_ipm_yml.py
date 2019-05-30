import cv2
import numpy as np
import os

def R_yaw(a):
    mat = np.zeros((3,3))
    mat[0,0] = np.cos(a)
    mat[0,1] = -np.sin(a)
    mat[1,0] = np.sin(a)
    mat[1,1] = np.cos(a)
    mat[2,2] = 1
    return mat

def R_pitch(a):
    mat = np.zeros((3,3))
    mat[0,0] = np.cos(a)
    mat[0,2] = np.sin(a)
    mat[1,1] = 1
    mat[2,0] = -np.sin(a)
    mat[2,2] = np.cos(a)
    return mat

def R_roll(a):
    mat = np.zeros((3,3))
    mat[0,0] = 1
    mat[1,1] = np.cos(a)
    mat[1,2] = -np.sin(a)
    mat[2,1] = np.sin(a)
    mat[2,2] = np.cos(a)
    return mat

def R(yaw, pitch, roll):
    mat = R_yaw(yaw).dot(R_pitch(pitch)).dot(R_roll(roll))
    return mat

def M_in(cx, cy, fx=None, fy=None, fov=None):
    mat = np.zeros((3,3))
    if fov is None:
        if fx is None and fy is None:
            fx = cx/2.0
            fy = fx
        mat[0, 0] = cx - 0.5
        mat[0, 1] = fx
        mat[1, 0] = cy - 0.5
        mat[1, 2] = fy
        mat[2, 0] = 1
    elif fx is None and fy is None:
        fx = cx / 2 / np.tan(fov/2/180*np.pi)
        fy = fx
        mat[0, 0] = cx - 0.5
        mat[0, 1] = fx
        mat[1, 0] = cy - 0.5
        mat[1, 2] = fy
        mat[2, 0] = 1
    else:
        raise Exception('Cannot give fx/fy and fov at the same time')
    return mat

class CamIntrExtr():
    def __init__(self, yaw, pitch, roll, height):
        self.R = R(yaw, pitch, roll)
        self.h = height

    def setIntr(self, cx, cy, fx=None, fy=None, fov = None):
        self.M_in = M_in(cx, cy, fx, fy, fov)

    def setBEV(self, x_end, y_width, u_bev=0, v_bev=0, x_start = 0, y_offset = 0):
        if u_bev == 0 and v_bev == 0:
            u_bev = y_width
            v_bev = x_end - x_start
        self.u_bev_width = u_bev
        self.v_bev_height = v_bev
        self.x_bev_long_dist = x_end - x_start
        self.x_bev_long_start = x_start
        self.x_bev_long_end = x_end
        self.y_bev_lat_dist = y_width
        self.y_bev_lat_start = y_offset - y_width/2
        self.y_bev_lat_end = y_offset + y_width/2

    def bev2world(self, us, vs):
        ys = (us+0.5) /self.u_bev_width * self.y_bev_lat_dist + self.y_bev_lat_start
        xs = (self.v_bev_height - (vs+0.5) ) / self.v_bev_height * self.x_bev_long_dist + self.x_bev_long_start
        return xs, ys
    
    def world2bev(self, xs, ys):
        us = (ys - self.y_bev_lat_start) / self.y_bev_lat_dist * self.u_bev_width - 0.5
        vs = self.v_bev_height - (xs - self.x_bev_long_start) / self.x_bev_long_dist * self.v_bev_height - 0.5
        return us, vs

    def world2img(self, xs, ys):
        # x front, y right, z down
        assert xs.shape == ys.shape, "x and y of of different shape"
        zs = np.ones_like(xs)*self.h
        coord_world = np.vstack((xs, ys, zs))
        coord_img = self.M_in.dot(self.R).dot(coord_world)
        us = coord_img[0, :] / coord_img[2, :]
        vs = coord_img[1, :] / coord_img[2, :]
        return us, vs
        
    def img2world(self, us, vs):
        assert us.shape == vs.shape, "u and v of of different shape"
        mat = np.linalg.inv( self.M_in.dot(self.R) )
        ts = np.ones_like(us)
        coord_img = np.vstack((us, vs, ts))
        coord_world_psudo = mat * coord_img
        z = coord_world_psudo[2, :]
        coord_world_psudo = coord_world_psudo / z * self.h
        xs = coord_world_psudo[0, :]
        ys = coord_world_psudo[1, :]
        return xs, ys

    def bev2img(self, u_bev, v_bev):
        x_world, y_world = self.bev2world(u_bev, v_bev)
        u_img, v_img = self.world2img(x_world, y_world)
        return u_img, v_img
    
    def img2bev(self, u_img, v_img):
        x_world, y_world = self.img2world(u_img, v_img)
        u_bev, v_bev = self.world2bev(x_world, y_world)
        return u_bev, v_bev


cam = CamIntrExtr(0, 10, 0, 2.2)
cam.setIntr(300, 150, fov = 70)
cam.setBEV(100, 30, 400, 600)
x_world = np.array([50, 100, 100, 50])
y_world = np.array([-10, -10, 10, 10])
u_img, v_img = cam.world2img(x_world, y_world)
u_bev, v_bev = cam.world2bev(x_world, y_world)

pts_img = np.vstack((u_img, v_img)).transpose((1, 0)).astype(np.float32)
pts_bev = np.vstack((u_bev, v_bev)).transpose((1, 0)).astype(np.float32)
print(pts_bev.shape)
print(pts_img.shape)

perspective_transform = cv2.getPerspectiveTransform(pts_img, pts_bev)

home = os.path.expanduser('~')
img_path = os.path.join(home, 'Carla/scenario_runner_cz/image_npy/imgs/cam_img084584.png')
img = cv2.imread(img_path)
bev = cv2.warpPerspective(img, perspective_transform, (400, 600))

for i in range(4):
    cv2.circle(img, tuple(pts_img[i]), 5, (0, 0, 255))
    cv2.circle(bev, tuple(pts_bev[i]), 5, (0, 0, 255))
cv2.imshow('img', img)
cv2.imshow('bev', bev)
cv2.waitKey(0)
