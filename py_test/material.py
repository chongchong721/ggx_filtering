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

import merl
import powit

import torch

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
        ndf = a_pow_of2 / (np.pi * np.power(cos_theta * cos_theta * (a_pow_of2-1) + 1,2))

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



class VNDFSamplerGGX:
    rng = None
    ggx = None

    def __init__(self, ax, ay, eta_i=1.0, eta_t=1.0, seed=int(time.time()), external_randomnumber=False, half_diff_pdf = False):
        """
        half_diff_pdf : if true, pdf will account for the Jacobian to theta_h,theta_d,phi_d. This is used if we want to
        importance sampling with respect to half-diff coordinate
        """
        #self.rng = np.random.default_rng(seed)
        self.ggx = GGX(ax, ay, eta_i, eta_t)
        self.half_diff_pdf = half_diff_pdf

    # The distribution of VNDF, is basically given you a theta_i,phi_i, what is the distribution of theta_m,phi_m
    # Distribution of anisotropic GGX:
    # https://www.pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models

    # https://computergraphics.stackexchange.com/questions/8555/anisotropic-ggx-brdf-implementation-how-is-it-related-to-isotropic-ggx-brdf
    # 1 / a^2 = cos^2{\phi} / ax^2 + sin^2{\phi} / ay^2 for isotropic case
    # Sample a GGX distribution of slopes with roughness(\alpha) = 1 given an incident cosine, we assume the phi is (x,0)
    # since we are sampling isotropic distribution, phi does not matter.
    def sample_ggx(self, theta_i):
        u, v = self.rng.random(2)

        # Special case ?
        if theta_i < 0.0001:
            r = np.sqrt(u / (1 - u))
            phi = np.pi * 2 * v
            return r * np.cos(phi), r * np.sin(phi)

        # Precomputation
        tan_theta = np.tan(theta_i)
        a = 1 / tan_theta
        g1 = 2 / (1 + np.sqrt(1 + 1.0 / (a * a)))

        # slope X
        A = 2 * u / g1 - 1
        tmp = 1.0 / (A * A - 1)
        B = tan_theta
        D = np.sqrt(B * B * tmp * tmp - (A * A - B * B) * tmp)
        slope_x_1 = B * tmp - D
        slope_x_2 = B * tmp + D
        if A < 0 or slope_x_2 > 1.0 / tan_theta:
            slope_x = slope_x_1
        else:
            slope_x = slope_x_2

        # sample y
        if v > 0.5:
            S = 1.0
            v = 2.0 * (v - 0.5)
        else:
            S = -1.0
            v = 2.0 * (0.5 - v)

        z = (v * (v * (v * 0.27385 - 0.73369) + 0.46341)) / (v * (v * (v * 0.093073 + 0.309420) - 1.000000) + 0.597999)
        slope_y = S * z * np.sqrt(1.0 + slope_x * slope_x)
        return slope_x, slope_y

    def slope_to_normal(self, slope_x, slope_y):
        norm = np.sqrt(slope_x * slope_x + slope_y * slope_y + 1)
        normal = np.array([[-slope_x.item()], [-slope_y.item()], [1.0]])
        normal = normal / norm
        return normal

    # A newer version of VNDF sampling
    # See https://jcgt.org/published/0007/04/01/
    def sample_anisotropic_ggx_new(self,wi, samples = None):
        ax = self.ggx.ax
        ay = self.ggx.ay

        if samples is None:
            u, v = self.rng.random(2)
        else:
            u, v = samples[0],samples[1]

        Vh = mat_util.makeVector(ax * wi[0][0],ay * wi[1][0],wi[2][0])
        Vh /= mat_util.norm_3D(Vh)

        lensq = Vh[0][0] ** 2 + Vh[1][0] ** 2

        if lensq > 0.0:
            T1 = mat_util.makeVector(-Vh[1][0],Vh[0][0],0)
            T1 = T1 / math.sqrt(lensq)
        else:
            T1 = mat_util.makeVector(1.0,0.0,0.0)

        #T2 = np.cross(Vh.T, T1.T).T

        T2 = mat_util.cross_3D(Vh,T1)

        r = np.sqrt(u)
        phi = np.pi * 2 * v

        t1 = r * math.cos(phi)
        t2 = r * math.sin(phi)

        s = 0.5 * (1.0 + Vh[2][0])
        t2 = (1.0 - s) * math.sqrt(1.0 - t1 ** 2) + s * t2

        Nh = t1 * T1 + t2 * T2 + math.sqrt(max(0.0,1.0-t1*t1-t2*t2)) * Vh

        Ne = mat_util.makeVector(ax * Nh[0][0],ay * Nh[1][0],max(0.0,Nh[2][0]))
        Ne /= mat_util.norm_3D(Ne)

        return Ne




    # Add stretch,rotation to make this ggx sample work in all cases
    def sample_anisotropic_ggx(self, wi):
        # stretch
        stretch_wi = np.array([[0.0], [0.0], [0.0]])
        stretch_wi[0] = self.ggx.ax * wi[0]
        stretch_wi[1] = self.ggx.ay * wi[1]
        stretch_wi[2] = wi[2]

        stretch_wi = stretch_wi / np.linalg.norm(stretch_wi)
        theta, phi = mat_util.to_spherical(stretch_wi)

        # sample
        slope_x, slope_y = self.sample_ggx(theta)

        # rotate
        tmp = np.cos(phi) * slope_x - np.sin(phi) * slope_y
        slope_y = np.sin(phi) * slope_x + np.cos(phi) * slope_y
        slope_x = tmp

        # unstretch
        slope_x = self.ggx.ax * slope_x
        slope_y = self.ggx.ay * slope_y

        return self.slope_to_normal(slope_x, slope_y)

    # According to adaptive paper.
    # w_i is sampled from the distribution of the VNDF. Need to sample theta/phi separately
    # theta_i is sampled from the distribution of projected area (marginal-conditional)
    # phi_i is sampled from distribution of VNDF where wi=(0,0,1)
    def sample_wi(self):
        wz = np.array([[0.0], [0.0], [1.0]])
        wm = self.sample_anisotropic_ggx(wz)

        # this is the phi for incident rays, also the phi we used to sample theta
        tmp, phi = mat_util.to_spherical(wm)

        u = self.rng.random()

        # cdf[idx-1] < u <= cdf[idx]
        idx = np.searchsorted(self.ggx.cdf, u, side='left')

        stride_theta = np.pi / 2 / len(self.ggx.cdf)

        if idx == 0:
            theta = 0
        else:
            a = self.ggx.cdf[idx - 1]
            b = self.ggx.cdf[idx]
            theta_a = stride_theta * idx
            theta_b = stride_theta * (idx + 1)
            theta = mat_util.lerp(theta_a, theta_b, (b - u) / (b - a))

        wi = mat_util.to_cartesian(theta, phi)
        return wi

    # pdf of sampling cos(theta)
    def wi_u_pdf(self,theta):
        u = np.cos(theta)
        stride_u = 1 / (len(self.ggx.cdf) - 1)
        left_idx = int(np.floor(u / stride_u))
        right_idx = int(np.ceil(u / stride_u))
        left = left_idx * stride_u
        right = right_idx * stride_u
        pdf_left = self.ggx.pdf[left_idx]
        pdf_right = self.ggx.pdf[right_idx]
        return mat_util.lerp(pdf_left,pdf_right,(right - u) / (right - left))

    # For isotropic cases only
    def sample_wi_u(self, samples = None):
        if samples is None:
            u, v = self.rng.random(2)
        else:
            u, v = samples[0],samples[1]
        phi = v * np.pi * 2
        stride_u = 1 / (len(self.ggx.cdf)-1)
        idx = np.searchsorted(self.ggx.cdf, u, side='left')

        if idx == 0:
            u_sample = 0
            pdf_sample = self.ggx.pdf[0]
        else:
            a = self.ggx.cdf[idx - 1]
            b = self.ggx.cdf[idx]
            pdf_a = self.ggx.pdf[idx - 1]
            pdf_b = self.ggx.pdf[idx]
            u_a = stride_u * (idx - 1)
            u_b = stride_u * idx
            u_sample = mat_util.lerp(u_a, u_b, (b - u) / (b - a))
            pdf_sample = mat_util.lerp(pdf_a,pdf_b,(b-u)/(b-a))

        theta = np.arccos(u_sample)

        #return mat_util.to_cartesian(theta,phi),pdf_sample * (1 / (2 * np.pi))
        return mat_util.to_cartesian(theta, phi)

    def sample(self):
        wi = self.sample_wi()
        wm = self.sample_anisotropic_ggx(wi)
        wo = mat_util.reflect(-wi, wm)
        return wi, wo

    def sample_wo(self, wi, samples = None):
        #wm = self.sample_anisotropic_ggx(wi)
        wm = self.sample_anisotropic_ggx_new(wi, samples)
        wo = mat_util.reflect(-wi, wm)

        pdf = self.ggx.vndf(wi, wm) * (0.25 / mat_util.dot(wo, wm))

        return wo, pdf

    def sample_wi_uniform(self, samples = None):
        if samples is None:
            u,v = self.rng.random(2)
        else:
            u,v = samples[0],samples[1]
        wi = mat_util.random_uniform_hemisphere(u,v)
        return wi

    def pdf(self, wi, wo, wm=None):
        if wm is None:
            wm = mat_util.get_half_vector(wi, wo)

        # Which one is correct?
        # return self.ggx.vndf(wi, wm) * (4 * mat_util.dot(wo, wm))
        return self.ggx.vndf(wi, wm) / (4 * mat_util.dot(wo, wm))


    def pdf_wi(self,wi):
        theta,phi = mat_util.to_spherical(wi)
        wi_pdf = self.wi_u_pdf(theta) * (1 / (2 * np.pi))
        return wi_pdf

    def pdf_wi_uniform(self,wi):
        return 1 / (np.pi * 2)

    def pdf_wo(self,wi,wo,wm):
        wo_pdf = self.ggx.vndf(wi, wm) / np.max([1e-6, (4 * math.fabs(mat_util.dot(wo, wm)))])
        return wo_pdf

    #pdf of perturbing both wi and wo
    def pdf_full(self,wi,wo,wm = None):
        if wm is None:
            wm = mat_util.get_half_vector(wi, wo)
        wo_pdf = self.pdf_wo(wi,wo,wm)
        wi_pdf = self.pdf_wi(wi)
        return wi_pdf * wo_pdf

    def pdf_full_uniform_wi(self,wi,wo,wm = None):
        if wm is None:
            wm = mat_util.get_half_vector(wi,wo)
        wo_pdf = self.pdf_wo(wi,wo,wm)
        wi_pdf = self.pdf_wi_uniform(wi)

        product = wo_pdf * wi_pdf

        if not self.half_diff_pdf:
            return product
        else:
            # 4 * cos(theta_d)
            cos_td = mat_util.dot(wo,wm)
            sin_td = np.sqrt(1 - cos_td ** 2)
            dwo_dwh = 4 * cos_td
            result = product * dwo_dwh * sin_td * np.sqrt(1 - wm[2][0]**2)
            return result

    def pdf_wh(self, wi, wm):
        return self.ggx.vndf(wi,wm)


class powit_merl_ndf:
    merl_tabular : powit.isotropic_merl_ndf

    #Not that, merl_ndf has a non-uniform resolution of 128
    # theta is parameterized

    def __init__(self, fielname, res = 128):
        """

        :param fielname:
        :param res:
        :param profile: 2 means smith model, do not change
        """
        self.torch_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.merl_tabular = powit.isotropic_merl_ndf(fielname, res)

        tmp_u = np.linspace(0, 1, res)


        self.merl_ndf_theta = tmp_u ** 2 * np.pi / 2
        self.merl_ndf_theta_torch = torch.from_numpy(self.merl_ndf_theta).to(self.torch_device)

        self.merl_ndf = np.array(self.merl_tabular.get_ndf())
        self.merl_ndf_torch = torch.from_numpy(self.merl_ndf).to(self.torch_device)

    def get_ndf_torch(self, cos_theta:torch.Tensor):
        """
        :param cos_theta:
        :return:
        """
        theta_value = torch.arccos(cos_theta)

        original_shape = theta_value.shape
        theta_value_flat = theta_value.view(-1)

        #all theta value should be within the range of 0,np.pi/2

        theta_idx = torch.searchsorted(self.merl_ndf_theta_torch, theta_value_flat, right=True)

        theta_0 = self.merl_ndf_theta_torch[theta_idx - 1]
        theta_1 = self.merl_ndf_theta_torch[theta_idx]

        f0 = self.merl_ndf_torch[theta_idx - 1]
        f1 = self.merl_ndf_torch[theta_idx]


        portion_to_theta0 = (theta_value_flat - theta_0) / (theta_1 - theta_0)

        ndf_flat = f1 * portion_to_theta0 + f0 * (1 - portion_to_theta0)

        ndf = ndf_flat.view(original_shape)

        return ndf

    def get_fitted_alpha(self):
        return self.merl_tabular.get_fitted_alpha()



class  MERL_MAT:
    """
    The MERL material uses theta_h,d,phi_d parameterization used by original MERL paper,
    I use a different one, will this cause problem?

    MERL returns non-cos weighted BRDF
    """
    mat : merl.MERL_BRDF

    def __init__(self,filename):
        self.mat = merl.MERL_BRDF(filename)

    # Create rotation matrices for rotations around x,y, and z axes.
    def RxMatrix(self,theta):
        return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

    def RyMatrix(self,theta):
        return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

    def RzMatrix(self,theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    # Convert rus-coords to two direction vectors
    # In:    Rusinkiewicz coordinates
    # Out:   Tuple of direction vectors (omega_o,omega_i)
    def RusinkToDirections(self, theta_h, theta_d, phi_d):
        # Initially put Halfvector along Z axis
        H = [0, 0, 1]
        omega_o = [np.sin(theta_d), 0, np.cos(theta_d)]
        omega_i = [-np.sin(theta_d), 0, np.cos(theta_d)]
        # Rotate phiD-pi/2 around the z-axis
        omega_o = np.dot(self.RzMatrix(phi_d - np.pi / 2), omega_o)
        omega_i = np.dot(self.RzMatrix(phi_d - np.pi / 2), omega_i)
        H = np.dot(self.RzMatrix(phi_d + np.pi / 2), H)
        # Rotate thetaH around x-axis
        omega_o = np.dot(self.RxMatrix(-theta_h), omega_o)
        omega_i = np.dot(self.RxMatrix(-theta_h), omega_i)
        H = np.dot(self.RxMatrix(-theta_h), H)
        # Rotate around z-axis so omega_o aligns with x-axis
        angl = -np.arccos(np.dot((1, 0, 0), self.normalize((omega_o[0], omega_o[1], 0)))) * np.sign(
            omega_o[1])  ##-omega_o[1]
        omega_o = np.dot(self.RzMatrix(angl), omega_o)
        omega_i = np.dot(self.RzMatrix(angl), omega_i)
        H = np.dot(self.RzMatrix(angl), H)

        return (omega_o, omega_i)

    def DirectionsToRusink(self,wi,wo):
        a = np.reshape(self.normalize(wi), (-1, 3))
        b = np.reshape(self.normalize(wo), (-1, 3))
        H = self.normalize((a + b) / 2)
        theta_h = np.arccos(H[:, 2])
        phi_h = np.arctan2(H[:, 1], H[:, 0])
        biNormal = np.array((0, 1, 0))
        normal = np.array((0, 0, 1))
        tmp = self.rotateVector(a, normal, -phi_h) # in or out? compare to cpp code
        diff = self.rotateVector(tmp, biNormal, -theta_h)
        diff = self.normalize(diff)
        theta_d = np.arccos(diff[:, 2])
        #phi_d = np.mod(np.arctan2(diff[:, 1], diff[:, 0]), np.pi)
        phi_d = np.arctan2(diff[:, 1], diff[:, 0])
        return np.column_stack((phi_d, theta_h, theta_d))

    def normalize(self, v: np.ndarray):
        return v / np.linalg.norm(v)

    # Rotate vector around arbitrary axis
    def rotateVector(self,vector, axis, angle):
        cos_ang = np.reshape(np.cos(angle), (-1))
        sin_ang = np.reshape(np.sin(angle), (-1))
        vector = np.reshape(vector, (-1, 3))
        axis = np.reshape(np.array(axis), (-1, 3))
        return vector * cos_ang[:, np.newaxis] + axis * np.dot(vector, np.transpose(axis)) * (1 - cos_ang)[:,
                                                                                             np.newaxis] + np.cross(
            axis, vector) * sin_ang[:, np.newaxis]

    def eval(self,wi,wo,wm):
        theta_i,phi_i = mat_util.to_spherical_negphi(wi)
        theta_o,phi_o = mat_util.to_spherical_negphi(wo)
        fr = self.mat.look_up(theta_i,phi_i,theta_o,phi_o)
        return fr[1]

    def eval_rgb_hd(self,theta_h,theta_d,phi_d):
        idx = merl.get_index_from_hall_diff_coords(theta_h, theta_d, phi_d)
        result = merl.get_half_diff_idxes_from_index(idx)
        fr = self.mat.look_up_hdidx(result[0],result[1],result[2])
        return fr

    def eval_rgb(self,wi,wo,wm):
        theta_i,phi_i = mat_util.to_spherical(wi)
        theta_o,phi_o = mat_util.to_spherical(wo)
        fr = self.mat.look_up(theta_i,phi_i,theta_o,phi_o)
        return np.asarray(fr)

    # Used in powit testing
    def eval_rgb_nocosweight(self,wi,wo,wm):
        result = self.eval_rgb(wi,wo,wm)
        result = result.reshape((3,1))
        return result


    def eval_channel(self,wi,wo,wh,channel_idx):
        theta_i,phi_i = mat_util.to_spherical_negphi(wi)
        theta_o,phi_o = mat_util.to_spherical_negphi(wo)
        fr = self.mat.look_up(theta_i,phi_i,theta_o,phi_o)
        return fr[channel_idx]



def rotate_vector_to_up(N, V, eps=1e-8):
    """
    Rotates vector V by the rotation that aligns N with the up vector (0,0,1).

    Args:
        N (np.ndarray): Normal vector of shape (3,).
        V (np.ndarray): Vector to be rotated of shape (3,).
        eps (float): Small epsilon value to handle numerical stability.

    Returns:
        np.ndarray: Rotated vector V' of shape (3,).
    """
    # Ensure N and V are numpy arrays
    N = np.asarray(N, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    # Normalize N and V
    N_norm = N / (np.linalg.norm(N) + eps)
    V_norm = V / (np.linalg.norm(V) + eps)

    # Target up vector
    T = np.array([0.0, 0.0, 1.0])

    # Compute cross product and dot product
    cross = np.cross(N_norm, T)
    cross_norm = np.linalg.norm(cross)

    dot = np.dot(N_norm, T)

    # Check if N is already aligned with T
    if cross_norm < eps:
        if dot > 0:
            # N is already aligned with T
            R = np.eye(3)
        else:
            # N is opposite to T, rotate 180 degrees around x-axis
            R = np.array([
                [1.0,  0.0,  0.0],
                [0.0, -1.0,  0.0],
                [0.0,  0.0, -1.0]
            ])
    else:
        # Compute rotation axis (k) and angle (theta)
        k = cross / cross_norm
        theta = np.arccos(np.clip(dot, -1.0, 1.0))

        # Skew-symmetric matrix K
        K = np.array([
            [0.0,    -k[2],  k[1]],
            [k[2],   0.0,   -k[0]],
            [-k[1],  k[0],   0.0]
        ])

        # Rodrigues' rotation formula
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    # Apply rotation
    V_rot = np.dot(R, V_norm)
    return V_rot




def rotate_vectors_to_up_batch(N_batch, V_batch, eps=1e-8):
    """
    Rotates a batch of vectors V_batch by the rotation that aligns each N_batch with the up vector (0,0,1).

    Args:
        N_batch (np.ndarray): Batch of normal vectors of shape (B, 3).
        V_batch (np.ndarray): Batch of vectors to be rotated of shape (B, 3).
        eps (float): Small epsilon value to handle numerical stability.

    Returns:
        np.ndarray: Rotated vectors V' of shape (B, 3).
    """
    # Ensure inputs are numpy arrays
    N_batch = np.asarray(N_batch, dtype=np.float64)
    V_batch = np.asarray(V_batch, dtype=np.float64)

    # Normalize N and V
    N_norm = N_batch / (np.linalg.norm(N_batch, axis=1, keepdims=True) + eps)
    V_norm = V_batch / (np.linalg.norm(V_batch, axis=1, keepdims=True) + eps)

    # Target up vector
    T = np.array([0.0, 0.0, 1.0])

    # Compute cross and dot products
    cross = np.cross(N_norm, T)  # Shape: (B, 3)
    cross_norm = np.linalg.norm(cross, axis=1)  # Shape: (B,)

    dot = np.einsum('ij,j->i', N_norm, T)  # Shape: (B,)

    # Initialize rotation matrices as identity
    B = N_batch.shape[0]
    R = np.tile(np.eye(3), (B, 1, 1))  # Shape: (B, 3, 3)

    # Handle cases where cross_norm is not near zero
    mask = cross_norm > eps  # Shape: (B,)

    if np.any(mask):
        # Rotation axis k and angle theta
        k = cross[mask] / cross_norm[mask, np.newaxis]  # Shape: (M, 3)
        theta = np.arccos(np.clip(dot[mask], -1.0, 1.0))  # Shape: (M,)

        # Skew-symmetric matrices K
        K = np.zeros((k.shape[0], 3, 3))
        K[:, 0, 1] = -k[:, 2]
        K[:, 0, 2] =  k[:, 1]
        K[:, 1, 0] =  k[:, 2]
        K[:, 1, 2] = -k[:, 0]
        K[:, 2, 0] = -k[:, 1]
        K[:, 2, 1] =  k[:, 0]

        # Compute Rodrigues' rotation matrices
        I = np.tile(np.eye(3), (k.shape[0], 1, 1))  # Shape: (M, 3, 3)
        R_rot = I + np.sin(theta)[:, np.newaxis, np.newaxis] * K + \
                (1 - np.cos(theta))[:, np.newaxis, np.newaxis] * np.matmul(K, K)

        # Assign rotation matrices
        R[mask] = R_rot

    # Handle cases where cross_norm is near zero
    mask_identity = ~mask  # Shape: (B,)
    if np.any(mask_identity):
        # For these cases, check if N is aligned or opposite to T
        # If dot >0: already aligned, R is identity
        # If dot <0: rotate 180 degrees around x-axis
        mask_opposite = (dot < -0.9999) & mask_identity  # Shape: (B,)
        if np.any(mask_opposite):
            # Define 180-degree rotation around x-axis
            R_opposite = np.array([
                [1.0,  0.0,  0.0],
                [0.0, -1.0,  0.0],
                [0.0,  0.0, -1.0]
            ])  # Shape: (3, 3)
            R[mask_opposite] = R_opposite

        # No action needed for already aligned vectors (R remains identity)

    # Apply rotation matrices to V
    V_rot = np.einsum('bij,bj->bi', R, V_norm)  # Shape: (B, 3)
    return V_rot


def K(brdf_val, o: np.array, h: np.array):
    """
    Compute K(o,h) in power iteration paper

    o,h are not in conventional shape..

    :param brdf: the material
    :param o: [3,1] both should be normalized
    :param h: [3,N] both should be normalized
    :return: K in shape (N,)
    """
    cos_theta_o = o[2,:]
    cos_theta_h = h[2,:]

    sec_theta_h = 1 / cos_theta_h

    dot_product = np.dot(h.T, o.flatten())

    k_array = 4 * brdf_val * np.power(cos_theta_o, 5) * np.power(sec_theta_h, 4) * dot_product

    return k_array

def integrate_K_prime(brdf_val, theta_o, theta_h, integrate_res, quadrature_weight):
    """
    \phi_o is does not matter, set to 0
    :param brdf:
    :param theta_o:
    :param theta_h:
    :param integrate_res:
    :return:
    """
    cos_theta_o = np.cos(theta_o)
    cos_theta_h = np.cos(theta_h)
    sin_theta_h = np.sin(theta_h)

    phi_h_list = np.linspace(0, np.pi * 2, integrate_res, endpoint=False)
    h_list = mat_util.to_cartesian_vectorized(theta_h,phi_h_list)

    o = mat_util.to_cartesian(theta_o, 0.0)

    #numerical compute the integral
    delta_s = np.pi * 2 / integrate_res

    integral = delta_s * quadrature_weight * sin_theta_h * np.sum(K(brdf_val, o, h_list))

    return integral


def build_powit_K(brdf:MERL_MAT, N):
    phi_o = 0

    u_array = np.linspace(0,1.0,N, endpoint=False)
    theta_o_array = u_array ** 2 * np.pi / 2
    o_array = mat_util.to_cartesian_vectorized(theta_o_array, phi_o)
    theta_h_array = np.copy(theta_o_array)

    K_matrix = np.zeros((N, N))

    # First implement a non-vectorized version
    for i in range(N):
        current_theta_o = theta_o_array[i]
        current_o = o_array[:,i].reshape((3,1))

        brdf_val = brdf.eval_rgb_nocosweight(current_o,current_o,current_o)
        brdf_green = brdf_val[1]

        for j in range(N):
            current_theta_h = theta_h_array[j]
            K_matrix[i,j] = integrate_K_prime(brdf_green, current_theta_o, current_theta_h, 128, 1 / N)


    p = np.ones((N,1))

    for i in range(4):
        p = np.matmul(K_matrix, p)

    #normalize p

    p = p.flatten()


    p_normalized = normalize_slope_p(p, N)

    d = p_normalized *  np.power((1/np.cos(theta_o_array)),4)

    return p_normalized



def normalize_slope_p(p, N):
    u_array = np.linspace(0, 1.0, N, endpoint=False)
    theta_o_array = u_array ** 2 * np.pi / 2
    tan_theta = np.tan(theta_o_array)
    sec_theta = 1 / np.cos(theta_o_array)
    du = 1 / N
    sum_of_p = np.sum(np.pi * 2 * p * np.pi * u_array * tan_theta * sec_theta * sec_theta * du)
    p_normalized = p / sum_of_p

    return p_normalized



def generate_all_merl_alpha():
    import os
    full_merl_directory = '/home/yuan/school/graphics/BRDFDatabase/brdfs/'
    for root, dirs, files in os.walk(full_merl_directory):
        print("Root:", root)
        print("Directories:", dirs)
        print("Files:", files)
        break

    dict_alpha = {}

    for file in files:
        if file[-7:] == '.binary':
            brdf_name = full_merl_directory + file
            tmp = powit_merl_ndf(brdf_name)
            dict_alpha[file[:-7]] = tmp.get_fitted_alpha()

    def write_dict_to_txt(dictionary, filename, delimiter=' '):
        """
        Writes a dictionary to a plain text file with each key-value pair on a separate line.

        Parameters:
        - dictionary (dict): The dictionary to write. Keys should be strings, and values should be floats.
        - filename (str): The path to the file where the dictionary will be saved.
        - delimiter (str): The string used to separate keys and values. Default is a space.
        """
        try:
            with open(filename, 'w') as file:
                for key, value in dictionary.items():
                    file.write(f"{key}{delimiter}{value}\n")
            print(f"Dictionary successfully saved to '{filename}'.")
        except IOError as e:
            print(f"An error occurred while writing to the file: {e}")


    def read_txt_to_dict(filename, delimiter=' '):
        """
        Reads a plain text file and converts it back into a dictionary.

        Parameters:
        - filename (str): The path to the file to read.
        - delimiter (str): The string used to separate keys and values. Must match the delimiter used during writing.

        Returns:
        - dict: The reconstructed dictionary.
        """
        reconstructed_dict = {}
        try:
            with open(filename, 'r') as file:
                for line in file:
                    key, value = line.strip().split(delimiter)
                    reconstructed_dict[key] = float(value)  # Convert value back to float
            print(f"Dictionary successfully loaded from '{filename}'.")
        except IOError as e:
            print(f"An error occurred while reading the file: {e}")
        except ValueError as ve:
            print(f"Value conversion error: {ve}")
        return reconstructed_dict

    # Sorting the Dictionary by Values (Low to High)
    sorted_dict_alpha = dict(sorted(dict_alpha.items(), key=lambda item: item[1]))


    write_dict_to_txt(sorted_dict_alpha, 'brdf_fitted_alpha.txt', delimiter=':')







def merl_test():
    test_merl_material = MERL_MAT("../merl_database/alum-bronze.binary")
    ndf = build_powit_K(test_merl_material,128)


def powit_test():
    test_powit_tab = powit_merl_ndf("../merl_database/alum-bronze.binary")


    theta = torch.from_numpy(np.random.random(20) * np.pi / 2)
    cos_theta = torch.cos(theta)

    test_powit_tab.get_ndf_torch(cos_theta)

if __name__ == "__main__":
    generate_all_merl_alpha()
    powit_test()
    merl_test()