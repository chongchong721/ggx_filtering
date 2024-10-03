"""
Material class
Direction should be in shape of [3,1] or [3,N]
here
"""
import builtins
from datetime import datetime

import numpy as np
import math
import mat_util
import time
import sys

class Lambertian:
    albedo = 1.0
    albedo_rgb : np.ndarray
    rng = None

    def __init__(self, a=1.0, rgb = np.array([1.0,1.0,1.0])):
        self.albedo = a
        self.albedo_rgb = rgb
        self.rng = np.random.default_rng(int(time.time()))

    # return cos-weighted lambertian brdf
    def eval(self, wi, wo, wm):
        if wo[2][0] < 0.0 or wi[2][0] < 0.0 or wm[2][0] < 0.0:
            return 0.0

        return self.albedo / np.pi * wo[2][0]

    # return cos-weighted channel brdf
    def eval_channel(self,wi,wo,wh,channel_idx):
        brdf = self.albedo_rgb[channel_idx] / np.pi * wo[2][0]
        return brdf

    def eval_vectorized(self,wi,wo,wm):
        """
        wi,wo,wm should be in the shape of [3,N]
        :param wi:
        :param wo:
        :param wm:
        :return:
        """
        brdf = self.albedo / np.pi * wo[2,:]
        brdf  = np.where(((wi[2,:] < 0)| (wo[2,:] < 0) | (wm[2,:] < 0)), 0.0 , brdf)
        return brdf

    def eval_vectorized_rgb_nocosweight(self,wi,wo,wm):
        r_channel = self.albedo_rgb[0] / np.pi
        r_channel  = np.where(((wi[2,:] < 0)| (wo[2,:] < 0) | (wm[2,:] < 0)), 0.0 , r_channel)
        g_channel = self.albedo_rgb[1] / np.pi
        g_channel  = np.where(((wi[2,:] < 0)| (wo[2,:] < 0) | (wm[2,:] < 0)), 0.0 , g_channel)
        b_channel = self.albedo_rgb[2] / np.pi
        b_channel  = np.where(((wi[2,:] < 0)| (wo[2,:] < 0) | (wm[2,:] < 0)), 0.0 , b_channel)
        return np.stack([r_channel,g_channel,b_channel])
    def eval_vectorized_rgb(self,wi,wo,wm):
        r_channel = self.albedo_rgb[0] / np.pi * wo[2,:]
        r_channel  = np.where(((wi[2,:] < 0)| (wo[2,:] < 0) | (wm[2,:] < 0)), 0.0 , r_channel)
        g_channel = self.albedo_rgb[1] / np.pi * wo[2,:]
        g_channel  = np.where(((wi[2,:] < 0)| (wo[2,:] < 0) | (wm[2,:] < 0)), 0.0 , g_channel)
        b_channel = self.albedo_rgb[2] / np.pi * wo[2,:]
        b_channel  = np.where(((wi[2,:] < 0)| (wo[2,:] < 0) | (wm[2,:] < 0)), 0.0 , b_channel)
        return np.stack([r_channel,g_channel,b_channel])

    def eval_rgb(self,wi,wo,wm):
        if wi[2][0] < 0.0 or wo[2][0] < 0.0:
            return mat_util.makeVector(0.0,0.0,0.0)
        else:
            result = self.albedo_rgb / np.pi * wo[2][0]
            result = result.reshape((3,1))
            return result

    def eval_rgb_nocosweighted(self,wi,wo,wm):
        if wi[2][0] < 0.0 or wo[2][0] < 0.0:
            return mat_util.makeVector(0.0,0.0,0.0)
        else:
            result = self.albedo_rgb / np.pi
            result = result.reshape((3,1))
            return result
    def sample_wo(self, wi):
        u, v = self.rng.random(2)
        wo = mat_util.random_uniform_hemisphere(u, v)
        return wo

    def pdf(self, wi, wo):
        return 1 / np.pi / 2




class GGX:
    ax = 0.0
    ay = 0.0
    cdf = None
    pdf = None
    eta_i = 1.0  # Outer surface's eta
    eta_t = 1.0
    albedo : np.ndarray

    ndf_list = []

    def __init__(self, ax, ay, eta_i=1.0, eta_t=1.0, albedo = np.array([1.0,1.0,1.0])):
        self.ax = ax
        self.ay = ay
        #self.generate_cdf()
        self.generate_cdf_u()
        self.eta_i = eta_i
        self.eta_t = eta_t
        self.albedo = albedo
        self.rng = np.random.default_rng(int(datetime.now().timestamp()))

    # Ramanujan's estimation of ellipse perimeter
    def ellipse_perimiter(self, a, b):
        h = ((a - b) * (a - b)) / ((a + b) * (a + b))
        csum = 1
        hp = 1
        hp *= h
        csum += (1.0 / 4.0) * hp
        hp *= h
        csum += (1.0 / 64.0) * hp
        hp *= h
        csum += (1.0 / 256.0) * hp
        hp *= h
        csum += (25.0 / 16384.0) * hp
        hp *= h
        csum += (49.0 / 65536.0) * hp
        hp *= h
        csum += (441.0 / 1048576.0) * hp
        return csum * np.pi * (a + b)

    def Lambda(self, w):
        theta, phi = mat_util.to_spherical(w)
        # Changing these two to safecos/safesin will cause problems at grazing angles?
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        alpha_tmp = np.sqrt(cos_phi * cos_phi * self.ax * self.ax + sin_phi * sin_phi * self.ay * self.ay)
        tan_theta = np.tan(theta)
        inverse_alpha = alpha_tmp * tan_theta
        return (-1.0 + np.sqrt(1.0 + inverse_alpha * inverse_alpha)) / 2.0

    def Lambda_vectorized(self,w):
        theta, phi = mat_util.to_spherical_vectorized(w)
        # Changing these two to safecos/safesin will cause problems at grazing angles?
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        alpha_tmp = np.sqrt(cos_phi * cos_phi * self.ax * self.ax + sin_phi * sin_phi * self.ay * self.ay)
        tan_theta = np.tan(theta)
        inverse_alpha = alpha_tmp * tan_theta
        return (-1.0 + np.sqrt(1.0 + inverse_alpha * inverse_alpha)) / 2.0

    # w is either wi or wo, wm is the normal
    def G1(self, w, wm):
        return mat_util.heaviside(w, wm) * 1 / (1 + self.Lambda(w))

    def G1_vectorized(self,w,wm):
        return mat_util.heaviside_vectorized(w,wm) * 1 / ( 1 + self.Lambda_vectorized(w))

    def G2(self, wi, wo, wm):
        return self.G1(wi, wm) * self.G1(wo, wm)

    def G2_vectorized(self,wi,wo,wm):
        return self.G1_vectorized(wi, wm) * self.G1_vectorized(wo, wm)
    # Sigma term used for VNDF normalization. See VNDF paper equation(5)
    def sigma(self, wi):
        theta, phi = mat_util.to_spherical(wi)
        cos_theta = np.cos(theta)
        return cos_theta * (1 + self.Lambda(wi))

    # Only isotropic for now
    # change variable from theta to cos(theta)
    def sigma_u(self, u):
        if u == 0.0:
            u = 0.001
        u_square = u**2
        #return u * np.sqrt(1-u_square) * (1 + np.sqrt(1 + self.ax * self.ax * (1 - u_square) / (u_square))) / 2
        return u * (1 + np.sqrt(1 + self.ax * self.ax * (1 - u_square) / (u_square))) / 2

    # Monte Carlo to estimate the sigma integral
    def estimate_sigma(self, wi):
        # Sample uniformlly from sphere
        sum = 0.0
        for i in range(100000):
            u, v = np.random.random(2)
            wm = mat_util.random_uniform_hemisphere(u, v)
            sum += mat_util.clamp_dot(wi, wm) * self.ndf(wm) * 2 * np.pi
        return sum / 100000.0

    # Sum of all sigma value where theta if fixed
    def marginal_sigma(self, theta):
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        tan_theta_square_inverse = cos_theta * cos_theta / sin_theta / sin_theta
        a = np.sqrt(self.ax * self.ax + tan_theta_square_inverse)
        b = np.sqrt(self.ay * self.ay + tan_theta_square_inverse)
        return cos_theta * np.pi + sin_theta / 2.0 * self.ellipse_perimiter(a, b)

    def estimate_marginal_sigma(self, theta):
        sum = 0.0
        for i in range(5000):
            phi = np.random.random(1) * 2 * np.pi
            w = mat_util.to_cartesian(theta, phi)
            sum += self.sigma(w) * 2 * np.pi
        return sum / 5000.0

    def generate_cdf_u(self,res=256):
        size = res
        self.cdf = np.zeros(size)
        self.pdf = np.zeros(size)
        stride_u = 1 / (size - 1)
        u = 0.0
        pdf = 0.0
        pdf_sum = 0.0
        for i in range(size):
            pdf = self.sigma_u(u)
            pdf_sum += pdf
            self.pdf[i] = pdf
            self.cdf[i] = pdf_sum
            u+=stride_u

        # Normalize the pdf
        integral = pdf_sum / size
        self.pdf = self.pdf / integral

        self.cdf = self.cdf / pdf_sum

    # It is not possible? to construct an inverse cdf to sample sigma, so we might use tabulated pdf
    def generate_cdf(self):
        size =4096
        self.cdf = np.zeros(size)
        stride_theta = np.pi / float(2 * size)
        theta = 0.0
        pdf = 0.0
        pdf_sum = 0.0
        for i in range(size):
            theta += stride_theta
            pdf = self.marginal_sigma(theta)
            pdf_sum += pdf
            self.cdf[i] = pdf_sum

        self.cdf = self.cdf / pdf_sum

    # Evaluate the specular BRDF at wi/wo/wm
    def eval(self, wi, wo, wm):
        if wi[2][0] < 0.0 or wo[2][0] < 0.0 or wm[2][0] < 0.0:
            return 0.0

        wg = np.array([[0.0], [0.0], [1.0]])
        denominator = 4 * math.fabs(mat_util.dot(wg, wo)) * math.fabs(mat_util.dot(wg, wi))
        G2 = self.G2(wi, wo, wm)
        # https://gamedev.stackexchange.com/questions/59758/microfacet-model-brdf-obtain-extreme-values-at-grazing-angles
        # if G2 < 0.0001:
        #     return 0.0
        # else:
        brdf = self.fresnel_term(wo, wm) * G2 * self.ndf(wm) / denominator
        # Return cos weighted GGX BRDF
        # return brdf * wo[2][0]
        # Return non-cos-weighted brdf
        return brdf

    def eval_vectorized(self, wi : np.ndarray, wo : np.ndarray, wm : np.ndarray):
        denominator = 4 * np.abs(wo[2,:]) * np.abs(wi[2,:])
        G2 = self.G2_vectorized(wi,wo,wm)

        fresnel_term = mat_util.fresnel_vectorized(wi,wm,1.0,self.eta_t)

        brdf = fresnel_term * G2 * self.ndf_vectorized(wm) / denominator

        brdf = np.where(((wi[2,:] < 0)| (wo[2,:] < 0) | (wm[2,:] < 0)), 0.0 , brdf)
        return brdf

    def eval_rgb(self, wi, wo, wm):
        result = self.eval(wi,wo,wm)
        r_channel = self.albedo[0] * result
        g_channel = self.albedo[1] * result
        b_channel = self.albedo[2] * result

        #return util.makeVector(r_channel,g_channel,b_channel)
        return np.array([r_channel,g_channel,b_channel])

    def eval_rgb_vectorized(self,wi,wo,wm):
        result = self.eval_vectorized(wi,wo,wm)
        r_channel = self.albedo[0] * result
        g_channel = self.albedo[1] * result
        b_channel = self.albedo[2] * result
        return np.stack([r_channel,g_channel,b_channel])

    # Used in powit testing
    def eval_rgb_nocosweight(self,wi,wo,wm):
        result = self.eval_rgb(wi,wo,wm)
        result = result.reshape((3,1))
        return result

    def eval_channel(self,wi,wo,wm,channel_idx):
        result = self.eval(wi,wo,wm)
        return self.albedo[channel_idx] * result


    # Fresnel dielectric
    def fresnel_term(self, wi, wm):
        if self.eta_t == 1.0 and self.eta_i == 1.0:
            return 1.0

        eta_i = self.eta_i
        eta_t = self.eta_t

        cos_theta_i = mat_util.dot(wi, wm)
        if cos_theta_i > 1.0:
            cos_theta_i = 1.0
        if cos_theta_i < 0:
            eta_i, eta_t = eta_t, eta_i
        sin_theta_i = np.sqrt(1 - cos_theta_i * cos_theta_i)
        sin_theta_t = eta_i / eta_t * sin_theta_i
        if sin_theta_t >= 1.0:
            return 1.0
        cos_theta_t = np.sqrt(1 - sin_theta_t * sin_theta_t)

        r_parl = ((eta_t * cos_theta_i) - (eta_i * cos_theta_t)) / ((eta_t * cos_theta_i) + (eta_i * cos_theta_t))
        r_perp = ((eta_i * cos_theta_i) - (eta_t * cos_theta_t)) / ((eta_i * cos_theta_i) + (eta_t * cos_theta_t))
        return (r_perp * r_perp + r_parl * r_parl) / 2

    # http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
    # which is equivalent to Heitz 14(Understanding Masking-shadowing function) equation(85)
    # assume geometric normal is (0,0,1)
    # return the NDF of an anisotropic ggx
    def ndf(self, wm):
        if wm[2][0] <= 0.0:
            return 0.0

        c = 1.0 / np.pi / self.ax / self.ay
        cos_g = mat_util.dot(np.array([[0.0], [0.0], [1.0]]), wm)

        inverse_term = 1.0 / (
                wm[0] / self.ax * wm[0] / self.ax + wm[1] / self.ay * wm[1] / self.ay + cos_g * cos_g).item()

        return c * inverse_term * inverse_term

    def ndf_vectorized(self,wm):
        c = 1.0 / np.pi / self.ax / self.ay
        cos_g = wm[2,:]
        inverse_term = 1.0 / (
                wm[0,:] * wm[0,:] / (self.ax * self.ax) + wm[1,:] * wm[1,:] / (self.ay * self.ay) + cos_g * cos_g)

        return c * inverse_term * inverse_term

    def ndf_isotropic(self, cos_theta):
        """
        Only cosine(theta) is needed in isotropic case
        :param cosine theta: computed directly from vector dot product
        :return:
        """
        assert self.ax == self.ay
        a = self.ax
        a_pow_of2 = a ** 2
        ndf = a_pow_of2 / (np.pi * np.pow(cos_theta * cos_theta * (a_pow_of2-1) + 1,2))

        ndf = np.where(cos_theta > 0.0, ndf, 0.0)

        return ndf

    # pdf of vndf
    def vndf(self, wi, wm):
        if wm[2][0] <= 0.0:
            return 0.0
        # wi[2][0] is the dot product of wi * geometric normal
        g1 = self.G1(wi, wm)
        ndf = self.ndf(wm)
        cos = math.fabs(mat_util.dot(wi, wm))
        return g1 * cos * ndf / np.max([wi[2][0],sys.float_info.epsilon])


    def sample_ndf(self,u = None, v = None, n_normals = 1):
        """
        only works for isotropic
        importance sampling ndf
        :param uv: uv random variables in shape[N,2]
        :param n_normals: [N,3]
        :return:
        """


        assert self.ax == self.ay

        if u is not None and v is not None:
            if n_normals > 1:
                assert u.szie == v.size
        else:
            u = self.rng.random(n_normals)
            v = self.rng.random(n_normals)

        theta = np.arctan(self.ax * np.sqrt(u) / np.sqrt(1-u))
        phi = np.pi * 2 * v

        if n_normals > 1:
            normals = mat_util.to_cartesian_vectorized(theta,phi)
            normals = normals.T
        else:
            normals = mat_util.to_cartesian(theta,phi)
            normals = normals.T

        return normals

    def sample_ndf_world(self,u = None, v = None, n_normals = 1, world_n = np.array([0,0,1])):
        #TODO check this is correct
        normals = self.sample_ndf(u, v, n_normals)
        # Convert this from local to world
        up_vector = np.zeros_like(normals)
        # find the face
        if world_n.max() == 1.0:
            idx = np.argmax(world_n)
        elif world_n.min() == -1.0:
            idx = np.argmin(world_n)
        else:
            raise NotImplementedError

        normals[:,idx] = 1.0

        world_n = np.tile(world_n, (n_normals, 1))
        tangentX = mat_util.normalized(np.cross(up_vector, world_n,axis=-1))
        tangentY = np.cross(world_n, tangentX, axis=-1)


        #compute world directions for samples
        samples = tangentX * np.tile(normals[:,0],(3,1)).T + \
                  tangentY * np.tile(normals[:,1],(3,1)).T + \
                  world_n * np.tile(normals[:,2],(3,1)).T
        return samples

    def ggx_mip_level(self, samples, world_n = np.array([0.0,0.0,1.0]), res = 128):
        """

        :param samples: sample location in [N,3]
        :param world_n: the world normal
        :param res: resolution of mipmap
        :return:
        """
        from map_util import jacobian_vertorized
        cosine = np.dot(samples, world_n)
        ndf = self.ndf_isotropic(cosine)
        sa_is = np.pi * 4 / res / 4.0 / ndf
        abs_max = np.max(np.abs(samples),axis=-1)
        samples_map = samples / np.stack((abs_max,abs_max,abs_max),axis=-1)
        sa_cube = 4 * jacobian_vertorized(samples_map) / (res ** 2)
        mip_level = 0.5 * (np.log2(sa_is/sa_cube))
        return mip_level
