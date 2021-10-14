import numpy as np
import cv2
import scipy.spatial.distance as ssd

#################################
# input regions from synth text
#################################


def filter_regions(regions, filt):
    """
    filt : boolean list of regions to keep.
    """
    idx = np.arange(len(filt))[filt]
    for k in regions.keys():
        regions[k] = [regions[k][i] for i in idx]
    return regions


def filter_for_placement(xyz, seg, regions):
    filt = np.zeros(len(regions['label'])).astype('bool')
    masks, Hs, Hinvs = [], [], []
    for idx, l in enumerate(regions['label']):
        res = get_text_placement_mask(
            xyz, seg == l, regions['coeff'][idx], pad=2)
        if res is not None:
            mask, H, Hinv = res
            masks.append(mask)
            Hs.append(H)
            Hinvs.append(Hinv)
            filt[idx] = True
    regions = filter_regions(regions, filt)
    regions['place_mask'] = masks
    regions['homography'] = Hs
    regions['homography_inv'] = Hinvs
    return regions


def rescale_frontoparallel(p_fp, box_fp, p_im):
    """
    The fronto-parallel image region is rescaled to bring it in
    the same approx. size as the target region size.

    p_fp : nx2 coordinates of countour points in the fronto-parallel plane
    box  : 4x2 coordinates of bounding box of p_fp
    p_im : nx2 coordinates of countour in the image

    NOTE : p_fp and p are corresponding, i.e. : p_fp[i] ~ p[i]

    Returns the scale 's' to scale the fronto-parallel points by.
    """
    l1 = np.linalg.norm(box_fp[1, :] - box_fp[0, :])
    l2 = np.linalg.norm(box_fp[1, :] - box_fp[2, :])

    n0 = np.argmin(np.linalg.norm(p_fp - box_fp[0, :][None, :], axis=1))
    n1 = np.argmin(np.linalg.norm(p_fp - box_fp[1, :][None, :], axis=1))
    n2 = np.argmin(np.linalg.norm(p_fp - box_fp[2, :][None, :], axis=1))

    lt1 = np.linalg.norm(p_im[n1, :] - p_im[n0, :])
    lt2 = np.linalg.norm(p_im[n1, :] - p_im[n2, :])

    s = max(lt1 / l1, lt2 / l2)
    if not np.isfinite(s):
        s = 1.0
    return s


def get_text_placement_mask(xyz, mask, plane, pad=2):
    """
    Returns a binary mask in which text can be placed.
    Also returns a homography from original image
    to this rectified mask.

    XYZ  : (HxWx3) image xyz coordinates
    MASK : (HxW) : non-zero pixels mark the object mask
    REGION : DICT output of TextRegions.get_regions
    PAD : number of pixels to pad the placement-mask by
    """
    contour, hier = cv2.findContours(mask.copy().astype('uint8'),
                                     mode=cv2.RETR_CCOMP,
                                     method=cv2.CHAIN_APPROX_SIMPLE)
    contour = [np.squeeze(c).astype('float') for c in contour]
    #plane = np.array([plane[1],plane[0],plane[2],plane[3]])
    H, W = mask.shape[:2]

    # bring the contour 3d points to fronto-parallel config:
    pts, pts_fp = [], []
    center = np.array([W, H]) / 2
    n_front = np.array([0.0, 0.0, -1.0])
    for i in range(len(contour)):
        cnt_ij = contour[i]
        xyz = DepthCamera.plane2xyz(center, cnt_ij, plane)
        R = rot3d(plane[:3], n_front)
        xyz = xyz.dot(R.T)
        pts_fp.append(xyz[:, :2])
        pts.append(cnt_ij)

    # unrotate in 2D plane:
    rect = cv2.minAreaRect(pts_fp[0].copy().astype('float32'))
    box = np.array(cv2.boxPoints(rect))
    R2d = unrotate2d(box.copy())
    box = np.vstack([box, box[0, :]])  # close the box for visualization

    mu = np.median(pts_fp[0], axis=0)
    pts_tmp = (pts_fp[0] - mu[None, :]).dot(R2d.T) + mu[None, :]
    boxR = (box - mu[None, :]).dot(R2d.T) + mu[None, :]

    # rescale the unrotated 2d points to approximately
    # the same scale as the target region:
    s = rescale_frontoparallel(pts_tmp, boxR, pts[0])
    boxR *= s
    for i in range(len(pts_fp)):
        pts_fp[i] = s * ((pts_fp[i] - mu[None, :]).dot(R2d.T) + mu[None, :])

    # paint the unrotated contour points:
    minxy = -np.min(boxR, axis=0) + pad // 2
    ROW = np.max(ssd.pdist(np.atleast_2d(boxR[:, 0]).T))
    COL = np.max(ssd.pdist(np.atleast_2d(boxR[:, 1]).T))

    place_mask = 255 * \
        np.ones((int(np.ceil(COL)) + pad, int(np.ceil(ROW)) + pad), 'uint8')

    pts_fp_i32 = [(pts_fp[i] + minxy[None, :]).astype('int32')
                  for i in range(len(pts_fp))]
    cv2.drawContours(place_mask, pts_fp_i32, -1, 0,
                     thickness=cv2.FILLED,
                     lineType=8, hierarchy=hier)

    if not TextRegions.filter_rectified((~place_mask).astype('float') / 255):
        return

    # calculate the homography
    H, _ = cv2.findHomography(pts[0].astype('float32').copy(),
                              pts_fp_i32[0].astype('float32').copy(),
                              method=0)

    Hinv, _ = cv2.findHomography(pts_fp_i32[0].astype('float32').copy(),
                                 pts[0].astype('float32').copy(),
                                 method=0)
    return place_mask, H, Hinv


class TextRegions(object):
    """
    Get region from segmentation which are good for placing
    text.
    """
    minWidth = 30  # px
    minHeight = 30  # px
    minAspect = 0.3  # w > 0.3*h
    maxAspect = 7
    minArea = 100  # number of pix
    pArea = 0.60  # area_obj/area_minrect >= 0.6

    # RANSAC planar fitting params:
    dist_thresh = 0.10  # m
    num_inlier = 90
    ransac_fit_trials = 100
    min_z_projection = 0.25

    minW = 20
    max_text_regions = 7

    @staticmethod
    def filter_rectified(mask):
        """
        mask : 1 where "ON", 0 where "OFF"
        """
        wx = np.median(np.sum(mask, axis=0))
        wy = np.median(np.sum(mask, axis=1))
        return wx > TextRegions.minW and wy > TextRegions.minW

    @staticmethod
    def get_hw(pt, return_rot=False):
        pt = pt.copy()
        R = unrotate2d(pt)
        mu = np.median(pt, axis=0)
        pt = (pt - mu[None, :]).dot(R.T) + mu[None, :]
        h, w = np.max(pt, axis=0) - np.min(pt, axis=0)
        if return_rot:
            return h, w, R
        return h, w

    @staticmethod
    def filter(seg, area, label):
        """
        Apply the filter.
        The final list is ranked by area.
        """
        good = label[area > TextRegions.minArea]
        area = area[area > TextRegions.minArea]
        filt, R = [], []
        for idx, i in enumerate(good):
            mask = seg == i
            xs, ys = np.where(mask)

            coords = np.c_[xs, ys].astype('float32')
            rect = cv2.minAreaRect(coords)
            box = np.array(cv2.boxPoints(rect))
            h, w, rot = TextRegions.get_hw(box, return_rot=True)

            f = (h > TextRegions.minHeight
                 and w > TextRegions.minWidth
                 and TextRegions.minAspect < w / h < TextRegions.maxAspect
                 and area[idx] / w * h > TextRegions.pArea)
            filt.append(f)
            R.append(rot)

        # filter bad regions:
        filt = np.array(filt)
        area = area[filt]
        R = [R[i] for i in range(len(R)) if filt[i]]

        # sort the regions based on areas:
        aidx = np.argsort(-area)
        good = good[filt][aidx]
        R = [R[i] for i in aidx]
        filter_info = {'label': good, 'rot': R, 'area': area[aidx]}
        return filter_info

    @staticmethod
    def sample_grid_neighbours(mask, nsample, step=3):
        """
        Given a HxW binary mask, sample 4 neighbours on the grid,
        in the cardinal directions, STEP pixels away.
        """
        if 2 * step >= min(mask.shape[:2]):
            return  # None

        y_m, x_m = np.where(mask)
        mask_idx = np.zeros_like(mask, 'int32')
        for i in range(len(y_m)):
            mask_idx[y_m[i], x_m[i]] = i

        xp, xn = np.zeros_like(mask), np.zeros_like(mask)
        yp, yn = np.zeros_like(mask), np.zeros_like(mask)
        xp[:, :-2 * step] = mask[:, 2 * step:]
        xn[:, 2 * step:] = mask[:, :-2 * step]
        yp[:-2 * step, :] = mask[2 * step:, :]
        yn[2 * step:, :] = mask[:-2 * step, :]
        valid = mask & xp & xn & yp & yn

        ys, xs = np.where(valid)
        N = len(ys)
        if N == 0:  # no valid pixels in mask:
            return  # None
        nsample = min(nsample, N)
        idx = np.random.choice(N, nsample, replace=False)
        # generate neighborhood matrix:
        # (1+4)x2xNsample (2 for y,x)
        xs, ys = xs[idx], ys[idx]
        s = step
        X = np.transpose(np.c_[xs, xs + s, xs + s, xs -
                               s, xs - s][:, :, None], (1, 2, 0))
        Y = np.transpose(np.c_[ys, ys + s, ys - s, ys +
                               s, ys - s][:, :, None], (1, 2, 0))
        sample_idx = np.concatenate([Y, X], axis=1)
        mask_nn_idx = np.zeros((5, sample_idx.shape[-1]), 'int32')
        for i in range(sample_idx.shape[-1]):
            mask_nn_idx[:, i] = mask_idx[sample_idx[:, :, i]
                                         [:, 0], sample_idx[:, :, i][:, 1]]
        return mask_nn_idx

    @staticmethod
    def filter_depth(xyz, seg, regions):
        plane_info = {'label': [],
                      'coeff': [],
                      'support': [],
                      'rot': [],
                      'area': []}
        for idx, l in enumerate(regions['label']):
            mask = seg == l
            pt_sample = TextRegions.sample_grid_neighbours(
                mask, TextRegions.ransac_fit_trials, step=3)
            if pt_sample is None:
                continue  # not enough points for RANSAC
            # get-depths
            pt = xyz[mask]
            plane_model = isplanar(pt, pt_sample,
                                   TextRegions.dist_thresh,
                                   TextRegions.num_inlier,
                                   TextRegions.min_z_projection)
            if plane_model is not None:
                plane_coeff = plane_model[0]
                if np.abs(plane_coeff[2]) > TextRegions.min_z_projection:
                    plane_info['label'].append(l)
                    plane_info['coeff'].append(plane_model[0])
                    plane_info['support'].append(plane_model[1])
                    plane_info['rot'].append(regions['rot'][idx])
                    plane_info['area'].append(regions['area'][idx])

        return plane_info

    @staticmethod
    def get_regions(xyz, seg, area, label):
        regions = TextRegions.filter(seg, area, label)
        # fit plane to text-regions:
        regions = TextRegions.filter_depth(xyz, seg, regions)
        return regions

    def get_num_text_regions(self, nregions):
        # return nregions
        nmax = min(self.max_text_regions, nregions)
        if np.random.rand() < 0.10:
            rnd = np.random.rand()
        else:
            rnd = np.random.beta(5.0, 1.0)
        return int(np.ceil(nmax * rnd))


class DepthCamera(object):
    """
    Camera functions for Depth-CNN camera.
    """
    f = 520

    @staticmethod
    def plane2xyz(center, ij, plane):
        """
        converts image pixel indices to xyz on the PLANE.

        center : 2-tuple
        ij : nx2 int array
        plane : 4-tuple

        return nx3 array.
        """
        ij = np.atleast_2d(ij)
        n = ij.shape[0]
        ij = ij.astype('float')
        xy_ray = (ij - center[None, :]) / DepthCamera.f
        z = -plane[2] / (xy_ray.dot(plane[:2]) + plane[3])
        xyz = np.c_[xy_ray, np.ones(n)] * z[:, None]
        return xyz

    @staticmethod
    def depth2xyz(depth):
        """
        Convert a HxW depth image (float, in meters)
        to XYZ (HxWx3).

        y is along the height.
        x is along the width.
        """
        H, W = depth.shape
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        X = (xx - W / 2) * depth / DepthCamera.f
        Y = (yy - H / 2) * depth / DepthCamera.f
        return np.dstack([X, Y, depth.copy()])

    @staticmethod
    def overlay(rgb, depth):
        """
        overlay depth and rgb images for visualization:
        """
        depth = depth - np.min(depth)
        depth /= np.max(depth)
        depth = (255 * depth).astype('uint8')
        return np.dstack([rgb[:, :, 0], depth, rgb[:, :, 1]])


def fit_plane(xyz, z_pos=None):
    """
    if z_pos is not None, the sign
    of the normal is flipped to make
    the dot product with z_pos (+).
    """
    mean = np.mean(xyz, axis=0)
    xyz_c = xyz - mean[None, :]
    l, v = np.linalg.eig(xyz_c.T.dot(xyz_c))
    abc = v[:, np.argmin(l)]
    d = -np.sum(abc * mean)
    # unit-norm the plane-normal:
    abcd = np.r_[abc, d] / np.linalg.norm(abc)
    # flip the normal direction:
    if z_pos is not None:
        if np.sum(abcd[:3] * z_pos) < 0.0:
            abcd *= -1
    return abcd


def fit_plane_ransac(pts, neighbors=None, z_pos=None, dist_inlier=0.05,
                     min_inlier_frac=0.60, nsample=3, max_iter=100):
    """
    Fits a 3D plane model using RANSAC.
    pts : (nx3 array) of point coordinates
    """
    n, _ = pts.shape
    ninlier, models = [], []
    for i in range(max_iter):
        if neighbors is None:
            p = pts[np.random.choice(pts.shape[0], nsample, replace=False), :]
        else:
            p = pts[neighbors[:, i], :]
        m = fit_plane(p, z_pos)
        ds = np.abs(pts.dot(m[:3]) + m[3])
        nin = np.sum(ds < dist_inlier)
        if nin / pts.shape[0] >= min_inlier_frac:
            ninlier.append(nin)
            models.append(m)

    if models == []:
        print("RANSAC plane fitting failed!")
        return  # None
    else:  # refit the model to inliers:
        ninlier = np.array(ninlier)
        best_model_idx = np.argsort(-ninlier)
        n_refit, m_refit, inliers = [], [], []
        for idx in best_model_idx[:min(10, len(best_model_idx))]:
            # re-estimate the model based on inliers:
            dists = np.abs(pts.dot(models[idx][:3]) + models[idx][3])
            inlier = dists < dist_inlier
            m = fit_plane(pts[inlier, :], z_pos)
            # compute new inliers:
            d = np.abs(pts.dot(m[:3]) + m[3])
            inlier = d < dist_inlier / 2  # heuristic
            n_refit.append(np.sum(inlier))
            m_refit.append(m)
            inliers.append(inlier)
        best_plane = np.argmax(n_refit)
        return m_refit[best_plane], inliers[best_plane]


def ensure_proj_z(plane_coeffs, min_z_proj):
    a, b, c, d = plane_coeffs
    if np.abs(c) < min_z_proj:
        s = ((1 - min_z_proj**2) / (a**2 + b**2))**0.5
        coeffs = np.array([s * a, s * b, np.sign(c) * min_z_proj, d])
        assert np.abs(np.linalg.norm(coeffs[:3]) - 1) < 1e-3
        return coeffs
    return plane_coeffs


def isplanar(xyz, sample_neighbors, dist_thresh, num_inliers, z_proj):
    """
    Checks if at-least FRAC_INLIERS fraction of points of XYZ (nx3)
    points lie on a plane. The plane is fit using RANSAC.

    XYZ : (nx3) array of 3D point coordinates
    SAMPLE_NEIGHBORS : 5xN_RANSAC_TRIALS neighbourhood array
                       of indices into the XYZ array. i.e. the values in this
                       matrix range from 0 to number of points in XYZ
    DIST_THRESH (default = 10cm): a point pt is an inlier iff dist(plane-pt)<dist_thresh
    FRAC_INLIERS : fraction of total-points which should be inliers to
                   to declare that points are planar.
    Z_PROJ : changes the surface normal, so that its projection on z axis is ATLEAST z_proj.

    Returns:
        None, if the data is not planar, else a 4-tuple of plane coeffs.
    """
    frac_inliers = num_inliers / xyz.shape[0]
    # align the normal to face towards camera
    dv = -np.percentile(xyz, 50, axis=0)
    max_iter = sample_neighbors.shape[-1]
    plane_info = fit_plane_ransac(xyz, neighbors=sample_neighbors,
                                  z_pos=dv, dist_inlier=dist_thresh,
                                  min_inlier_frac=frac_inliers, nsample=20,
                                  max_iter=max_iter)
    if plane_info is not None:
        coeff, inliers = plane_info
        coeff = ensure_proj_z(coeff, z_proj)
        return coeff, inliers
    else:
        return  # None


def ssc(v):
    """
    Returns the skew-symmetric cross-product matrix corresponding to v.
    """
    v /= np.linalg.norm(v)
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def rot3d(v1, v2):
    """
    Rodrigues formula : find R_3x3 rotation matrix such that v2 = R*v1.
    https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula#Matrix_notation
    """
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    v3 = np.cross(v1, v2)
    s = np.linalg.norm(v3)
    c = v1.dot(v2)
    Vx = ssc(v3)
    return np.eye(3) + s * Vx + (1 - c) * Vx.dot(Vx)


def unrotate2d(pts):
    """
    PTS : nx3 array
    finds principal axes of pts and gives a rotation matrix (2d)
    to realign the axes of max variance to x,y.
    """
    mu = np.median(pts, axis=0)
    pts -= mu[None, :]
    l, R = np.linalg.eig(pts.T.dot(pts))
    R = R / np.linalg.norm(R, axis=0)[None, :]

    # make R compatible with x-y axes:
    if abs(R[0, 0]) < abs(R[0, 1]):  # compare dot-products with [1,0].T
        R = np.fliplr(R)
    if not np.allclose(np.linalg.det(R), 1):
        if R[0, 0] < 0:
            R[:, 0] *= -1
        elif R[1, 1] < 0:
            R[:, 1] *= -1
        else:
            print("Rotation matrix not understood")
            return
    if R[0, 0] < 0 and R[1, 1] < 0:
        R *= -1
    assert np.allclose(np.linalg.det(R), 1)

    # at this point "R" is a basis for the original (rotated) points.
    # we need to return the inverse to "unrotate" the points:
    return R.T  # return the inverse
