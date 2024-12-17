// Copyright 2005 Mitsubishi Electric Research Laboratories All Rights Reserved.

// Permission to use, copy and modify this software and its documentation without
// fee for educational, research and non-profit purposes, is hereby granted, provided
// that the above copyright notice and the following three paragraphs appear in all copies.

// To request permission to incorporate this software into commercial products contact:
// Vice President of Marketing and Business Development;
// Mitsubishi Electric Research Laboratories (MERL), 201 Broadway, Cambridge, MA 02139 or
// <license@merl.com>.

// IN NO EVENT SHALL MERL BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL,
// OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND
// ITS DOCUMENTATION, EVEN IF MERL HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

// MERL SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED
// HEREUNDER IS ON AN "AS IS" BASIS, AND MERL HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
// UPDATES, ENHANCEMENTS OR MODIFICATIONS.

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <iostream>
#include <string>
#include <fstream>

#include <vector>

#include <random>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#define BRDF_SAMPLING_RES_THETA_H       90
#define BRDF_SAMPLING_RES_THETA_D       90
#define BRDF_SAMPLING_RES_PHI_D         360


#define M_PI	3.1415926535897932384626433832795


auto global_uniform_distribution = std::uniform_real_distribution<double>(0, 1);
auto rng = std::mt19937(std::random_device{}());


/* Standalone vec3 utility */
struct vec3 {
	static vec3 from_raw(const double *v) {return vec3((double)v[0], (double)v[1], (double)v[2]);}
	static vec3 from_raw(const float *v) {return vec3((double)v[0], (double)v[1], (double)v[2]);}
	static const double *to_raw(const vec3& v) {return &v.x;}
	explicit vec3(double x = 0): x(x), y(x), z(x) {}
	vec3(double x, double y, double z) : x(x), y(y), z(z) {}
	explicit vec3(double theta, double phi);
	double intensity() const {return (double)0.2126 * x + (double)0.7152 * y + (double)0.0722 * z;}
	double x, y, z;
};


vec3 random_hemisphere() {
	double u1 = global_uniform_distribution(rng);
	double u2 = global_uniform_distribution(rng);

	double phi = 2.0 * M_PI * u1;          // Phi in [0, 2pi)
	double cos_theta = u2;                 // Cos(theta) in [0,1]
	double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta); // Sin(theta)

	// Convert spherical coordinates to Cartesian coordinates
	double x = sin_theta * std::cos(phi);
	double y = sin_theta * std::sin(phi);
	double z = cos_theta; // Since we're sampling the upper hemisphere (z >= 0)

	return vec3(x, y, z);
}



class MERL_BRDF{
	public:

		const float RED_SCALE = 1.0 / 1500;
		const float GREEN_SCALE =  (1.15/1500.0);
		const float BLUE_SCALE  = (1.66/1500.0);



		// data has 90 * 90 * 180

		const uint32_t size = BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D * BRDF_SAMPLING_RES_PHI_D / 2;

		double * data;

		std::vector<double> vector_data_r;
		std::vector<double> vector_data_g;
		std::vector<double> vector_data_b;

		MERL_BRDF(std::string);


				// Lookup theta_half index
		// This is a non-linear mapping!
		// In:  [0 .. pi/2]
		// Out: [0 .. 89]
		inline int theta_half_index(double theta_half)
		{
			if (theta_half <= 0.0)
				return 0;
			double theta_half_deg = ((theta_half / (M_PI/2.0))*BRDF_SAMPLING_RES_THETA_H);
			double temp = theta_half_deg*BRDF_SAMPLING_RES_THETA_H;
			temp = sqrt(temp);
			int ret_val = (int)temp;
			if (ret_val < 0) ret_val = 0;
			if (ret_val >= BRDF_SAMPLING_RES_THETA_H)
				ret_val = BRDF_SAMPLING_RES_THETA_H-1;
			return ret_val;
		}


		// Lookup theta_diff index
		// In:  [0 .. pi/2]
		// Out: [0 .. 89]
		inline int theta_diff_index(double theta_diff)
		{
			int tmp = int(theta_diff / (M_PI * 0.5) * BRDF_SAMPLING_RES_THETA_D);
			if (tmp < 0)
				return 0;
			else if (tmp < BRDF_SAMPLING_RES_THETA_D - 1)
				return tmp;
			else
				return BRDF_SAMPLING_RES_THETA_D - 1;
		}


		// Lookup phi_diff index
		inline int phi_diff_index(double phi_diff)
		{
			// Because of reciprocity, the BRDF is unchanged under
			// phi_diff -> phi_diff + M_PI
			if (phi_diff < 0.0)
				phi_diff += M_PI;

			// In: phi_diff in [0 .. pi]
			// Out: tmp in [0 .. 179]
			int tmp = int(phi_diff / M_PI * BRDF_SAMPLING_RES_PHI_D / 2);
			if (tmp < 0)
				return 0;
			else if (tmp < BRDF_SAMPLING_RES_PHI_D / 2 - 1)
				return tmp;
			else
				return BRDF_SAMPLING_RES_PHI_D / 2 - 1;
		}



		// cross product of two vectors
		void cross_product (double* v1, double* v2, double* out);


		// normalize vector
		void normalize(double* v);

		// rotate vector along one axis
		void rotate_vector(double* vector, double* axis, double angle, double* out);



		// convert standard coordinates to half vector/difference vector coordinates
		void std_coords_to_half_diff_coords(double theta_in, double fi_in, double theta_out, double fi_out,
										double& theta_half,double& fi_half,double& theta_diff,double& fi_diff );






		// Given a pair of incoming/outgoing angles, look up the BRDF.
		void lookup_brdf_val(double theta_in, double fi_in,
					double theta_out, double fi_out,
					double& red_val,double& green_val,double& blue_val);

		// Read BRDF data
		bool read_brdf(const char *filename, double* &brdf);





		std::vector<double> lookup_wrapper(double theta_in,double fi_in,
					double theta_out, double fi_out);

		std::vector<double> lookup_wrapper_hdidxes(double theta_half,
					double theta_diff, double phi_diff);


		std::vector<double> half_diff_conversion_wrapper(double theta_in, double fi_in, double theta_out, double fi_out);

		double lookup_one_channel(double theta_in,double fi_in,
					double theta_out, double fi_out,int idx);


		void integrate_over_view_direction_monte_carlo(double theta_v,double &r_int, double &g_int, double &b_int, int n_it);
		std::vector<std::vector<double>> integrate_all_view_theta_monte_carlo(int view_cos_theta_resolution, int is_n_sample);

		void integrate_over_view_direction(double theta_v,double &r_int, double &g_int, double &b_int, int delta_factor);
		std::vector<std::vector<double>> integrate_all_view_theta(int view_cos_theta_resolution, int delta_factor);

};



MERL_BRDF::MERL_BRDF(std::string filename){
	if (!read_brdf(filename.c_str(), data))
	{
		fprintf(stderr, "Error reading %s\n", filename.c_str());
		std::runtime_error("Error reading BRDF");
	}else{
		vector_data_r = std::vector<double>(data,data + size);
		vector_data_g = std::vector<double>(data + size, data + size * 2);
		vector_data_b = std::vector<double>(data + size * 2, data + size * 3);
		//it takes too much memory
		free(this->data);
	}
}



// cross product of two vectors
void MERL_BRDF::cross_product (double* v1, double* v2, double* out)
{
	out[0] = v1[1]*v2[2] - v1[2]*v2[1];
	out[1] = v1[2]*v2[0] - v1[0]*v2[2];
	out[2] = v1[0]*v2[1] - v1[1]*v2[0];
}

// normalize vector
void MERL_BRDF::normalize(double* v)
{
	// normalize
	double len = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
	v[0] = v[0] / len;
	v[1] = v[1] / len;
	v[2] = v[2] / len;
}

// rotate vector along one axis
void MERL_BRDF::rotate_vector(double* vector, double* axis, double angle, double* out)
{
	double temp;
	double cross[3];
	double cos_ang = cos(angle);
	double sin_ang = sin(angle);

	out[0] = vector[0] * cos_ang;
	out[1] = vector[1] * cos_ang;
	out[2] = vector[2] * cos_ang;

	temp = axis[0]*vector[0]+axis[1]*vector[1]+axis[2]*vector[2];
	temp = temp*(1.0-cos_ang);

	out[0] += axis[0] * temp;
	out[1] += axis[1] * temp;
	out[2] += axis[2] * temp;

	cross_product (axis,vector,cross);

	out[0] += cross[0] * sin_ang;
	out[1] += cross[1] * sin_ang;
	out[2] += cross[2] * sin_ang;
}


// convert standard coordinates to half vector/difference vector coordinates
void MERL_BRDF::std_coords_to_half_diff_coords(double theta_in, double fi_in, double theta_out, double fi_out,
								double& theta_half,double& fi_half,double& theta_diff,double& fi_diff )
{

	// compute in vector
	double in_vec_z = cos(theta_in);
	double proj_in_vec = sin(theta_in);
	double in_vec_x = proj_in_vec*cos(fi_in);
	double in_vec_y = proj_in_vec*sin(fi_in);
	double in[3]= {in_vec_x,in_vec_y,in_vec_z};
	normalize(in);


	// compute out vector
	double out_vec_z = cos(theta_out);
	double proj_out_vec = sin(theta_out);
	double out_vec_x = proj_out_vec*cos(fi_out);
	double out_vec_y = proj_out_vec*sin(fi_out);
	double out[3]= {out_vec_x,out_vec_y,out_vec_z};
	normalize(out);


	// compute halfway vector
	double half_x = (in_vec_x + out_vec_x)/2.0f;
	double half_y = (in_vec_y + out_vec_y)/2.0f;
	double half_z = (in_vec_z + out_vec_z)/2.0f;
	double half[3] = {half_x,half_y,half_z};
	normalize(half);

	// compute  theta_half, fi_half
	theta_half = acos(half[2]);
	fi_half = atan2(half[1], half[0]);


	double bi_normal[3] = {0.0, 1.0, 0.0};
	double normal[3] = { 0.0, 0.0, 1.0 };
	double temp[3];
	double diff[3];

	// compute diff vector
	rotate_vector(in, normal , -fi_half, temp);
	rotate_vector(temp, bi_normal, -theta_half, diff);

	// compute  theta_diff, fi_diff
	theta_diff = acos(diff[2]);
	fi_diff = atan2(diff[1], diff[0]);

}





// Given a pair of incoming/outgoing angles, look up the BRDF.
void MERL_BRDF::lookup_brdf_val(double theta_in, double fi_in,
			double theta_out, double fi_out,
			double& red_val,double& green_val,double& blue_val)
{
	// Convert to halfangle / difference angle coordinates
	double theta_half, fi_half, theta_diff, fi_diff;

	std_coords_to_half_diff_coords(theta_in, fi_in, theta_out, fi_out,
			theta_half, fi_half, theta_diff, fi_diff);


	// Find index.
	// Note that phi_half is ignored, since isotropic BRDFs are assumed
	int ind = phi_diff_index(fi_diff) +
		theta_diff_index(theta_diff) * BRDF_SAMPLING_RES_PHI_D / 2 +
		theta_half_index(theta_half) * BRDF_SAMPLING_RES_PHI_D / 2 *
							BRDF_SAMPLING_RES_THETA_D;

	// red_val = this->data[ind] * RED_SCALE;
	// green_val = this->data[ind + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D/2] * GREEN_SCALE;
	// blue_val = this->data[ind + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D] * BLUE_SCALE;


	red_val = this->vector_data_r[ind] * RED_SCALE;
	green_val = this->vector_data_g[ind] * GREEN_SCALE;
	blue_val = this->vector_data_b[ind] * BLUE_SCALE;

	if (red_val < 0.0 || green_val < 0.0 || blue_val < 0.0)
		fprintf(stderr, "Below horizon.\n");

}



// Read BRDF data
bool MERL_BRDF::read_brdf(const char *filename, double* &brdf)
{
	FILE *f = fopen(filename, "rb");
	if (!f)
		return false;

	int dims[3];
	fread(dims, sizeof(int), 3, f);
	int n = dims[0] * dims[1] * dims[2];
	if (n != BRDF_SAMPLING_RES_THETA_H *
		BRDF_SAMPLING_RES_THETA_D *
		BRDF_SAMPLING_RES_PHI_D / 2)
	{
		fprintf(stderr, "Dimensions don't match\n");
		fclose(f);
		return false;
	}

	brdf = (double*) malloc (sizeof(double)*3*n);
	fread(brdf, sizeof(double), 3*n, f);

	fclose(f);
	return true;
}






std::vector<double> MERL_BRDF::lookup_wrapper(double theta_in,double fi_in, double theta_out, double fi_out){
	double red,green,blue;

	lookup_brdf_val(theta_in,fi_in,theta_out,fi_out,red,green,blue);

	return std::vector<double>{red,green,blue};

}


std::vector<double> MERL_BRDF::lookup_wrapper_hdidxes(double theta_half_idx, double theta_diff_idx, double phi_diff_idx){
	int ind = phi_diff_idx +
		theta_diff_idx * BRDF_SAMPLING_RES_PHI_D / 2 +
		theta_half_idx * BRDF_SAMPLING_RES_PHI_D / 2 *
							BRDF_SAMPLING_RES_THETA_D;

	// red_val = this->data[ind] * RED_SCALE;
	// green_val = this->data[ind + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D/2] * GREEN_SCALE;
	// blue_val = this->data[ind + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D] * BLUE_SCALE;


	double red_val = this->vector_data_r[ind] * RED_SCALE;
	double green_val = this->vector_data_g[ind] * GREEN_SCALE;
	double blue_val = this->vector_data_b[ind] * BLUE_SCALE;

	if (red_val < 0.0 || green_val < 0.0 || blue_val < 0.0)
		fprintf(stderr, "Below horizon.\n");


	return std::vector<double>{red_val,green_val,blue_val};
}



std::vector<double> MERL_BRDF::half_diff_conversion_wrapper(double theta_in, double fi_in, double theta_out, double fi_out){
	double theta_half, fi_half, theta_diff, fi_diff;

	std_coords_to_half_diff_coords(theta_in,fi_in,theta_out,fi_out,theta_half,fi_half,theta_diff,fi_diff);

	return std::vector<double>{theta_half,fi_half,theta_diff,fi_diff};

}




double MERL_BRDF::lookup_one_channel(double theta_in,double fi_in, double theta_out, double fi_out, int idx){
	std::shared_ptr<std::vector<double>> value;
	float scale;


	switch (idx)
	{
	case 0:
		value = std::make_shared<std::vector<double>>(this->vector_data_r);
		scale = RED_SCALE;
		break;
	case 1:
		value = std::make_shared<std::vector<double>>(this->vector_data_g);
		scale = GREEN_SCALE;
		break;
	case 2:
		value = std::make_shared<std::vector<double>>(this->vector_data_b);
		scale = BLUE_SCALE;
		break;

	default:
		std::runtime_error("Unsupported Index");
		break;
	}

	double theta_half, fi_half, theta_diff, fi_diff;

	std_coords_to_half_diff_coords(theta_in, fi_in, theta_out, fi_out,
			theta_half, fi_half, theta_diff, fi_diff);


	// Find index.
	// Note that phi_half is ignored, since isotropic BRDFs are assumed
	int ind = phi_diff_index(fi_diff) +
		theta_diff_index(theta_diff) * BRDF_SAMPLING_RES_PHI_D / 2 +
		theta_half_index(theta_half) * BRDF_SAMPLING_RES_PHI_D / 2 *
							BRDF_SAMPLING_RES_THETA_D;


	return value->at(ind) * scale;
}
inline void MERL_BRDF::integrate_over_view_direction(double theta_v, double &r_int, double &g_int, double &b_int, int delta_factor) {
	/**
	 *
	 * The default resolution is 1 degree. Which means, delta = PI / 180.0. If delta_factor > 1. It means we have a higher resolution
	 * delta = PI / 180 / delta_factor
	 **/

	const double phi_v = 0;

	double delta = M_PI / 180.0 / delta_factor; //more precision

	double delta_half = 0.5 * delta;

	double theta_l = delta_half;
	double phi_l = delta_half;

	double r_sum = 0.0;
	double g_sum = 0.0;
	double b_sum = 0.0;

	double r_channel = 0.0;
	double g_channel = 0.0;
	double b_channel = 0.0;

	int n_theta_loop = (int)((M_PI / 2) / (delta));
	int n_phi_loop = (int)((M_PI *2) / (delta));


	for(int i = 0; i < n_theta_loop; i++) {
		double sin_theta_l = sin(theta_l);
		double cos_theta_l = cos(theta_l);
		double sincos_l = sin_theta_l * cos_theta_l;

		for(int j = 0; j < n_phi_loop; j++) {
			this->lookup_brdf_val(theta_v,phi_v,theta_l,phi_l,
				r_channel,g_channel,b_channel);
			r_sum += (r_channel * sincos_l);
			g_sum += (g_channel * sincos_l);
			b_sum += (b_channel * sincos_l);

			phi_l += delta;
		}

		theta_l += delta;

		phi_l = delta_half; //set phi back to 0.0
	}

	r_int = r_sum * delta * delta;
	g_int = g_sum * delta * delta;
	b_int = b_sum * delta * delta;

}


std::vector<std::vector<double>> MERL_BRDF::integrate_all_view_theta(int view_cos_theta_resolution, int delta_factor) {
	std::vector<std::vector<double>> result;
	result.emplace_back();
	result.emplace_back();
	result.emplace_back();

	double delta = 1.0 / (view_cos_theta_resolution - 1);

	double cos_theta_v = 0.0;

	for(int i =0; i < view_cos_theta_resolution; i++) {
		double theta_v = acos(cos_theta_v);

		double r_result,g_result,b_result;

		this->integrate_over_view_direction(theta_v,r_result,g_result,b_result, delta_factor);

		result[0].push_back(r_result);
		result[1].push_back(g_result);
		result[2].push_back(b_result);

		cos_theta_v += delta;
	}

	return result;

}

inline void MERL_BRDF::integrate_over_view_direction_monte_carlo(double theta_v, double &r_int, double &g_int, double &b_int, int n_it) {
	double r_sum = 0.0;
	double g_sum = 0.0;
	double b_sum = 0.0;

	double r_channel = 0.0;
	double g_channel = 0.0;
	double b_channel = 0.0;

	const double phi_v = 0.0;

	for (int i = 0; i < n_it; i++) {
		auto direction_l = random_hemisphere();
		double cos_theta_l = direction_l.z;
		double theta_l = acos(cos_theta_l);
		double phi_l = atan2(direction_l.y, direction_l.x);


		this->lookup_brdf_val(theta_v,phi_v,theta_l,phi_l,
				r_channel,g_channel,b_channel);
		r_sum += (r_channel * cos_theta_l);
		g_sum += (g_channel * cos_theta_l);
		b_sum += (b_channel * cos_theta_l);

	}

	r_int = r_sum * M_PI * 2 / static_cast<double>(n_it);
	g_int = g_sum * M_PI * 2 / static_cast<double>(n_it);
	b_int = b_sum * M_PI * 2 / static_cast<double>(n_it);
}


inline std::vector<std::vector<double> > MERL_BRDF::integrate_all_view_theta_monte_carlo(int view_cos_theta_resolution, int is_n_sample) {
	std::vector<std::vector<double>> result;
	result.emplace_back();
	result.emplace_back();
	result.emplace_back();

	double delta = 1.0 / (view_cos_theta_resolution - 1);

	double cos_theta_v = 0.0;

	for(int i =0; i < view_cos_theta_resolution; i++) {
		double theta_v = acos(cos_theta_v);

		double r_result,g_result,b_result;

		this->integrate_over_view_direction_monte_carlo(theta_v,r_result,g_result,b_result, is_n_sample);

		result[0].push_back(r_result);
		result[1].push_back(g_result);
		result[2].push_back(b_result);

		cos_theta_v += delta;
	}

	return result;
}



double  reduce_fi_d(double phi_d){
	if (phi_d < 0.0)
		phi_d += M_PI;

	return phi_d;
}


inline int theta_half_index(double theta_half)
{
	if (theta_half <= 0.0)
		return 0;
	double theta_half_deg = ((theta_half / (M_PI/2.0))*BRDF_SAMPLING_RES_THETA_H);
	double temp = theta_half_deg*BRDF_SAMPLING_RES_THETA_H;
	temp = sqrt(temp);
	int ret_val = (int)temp;
	if (ret_val < 0) ret_val = 0;
	if (ret_val >= BRDF_SAMPLING_RES_THETA_H)
		ret_val = BRDF_SAMPLING_RES_THETA_H-1;
	return ret_val;
}


inline double theta_half_rad(int idx){
	double temp = idx * idx;
	temp /= BRDF_SAMPLING_RES_THETA_H;

	return temp * M_PI / 180.0;
}


// Lookup theta_diff index
// In:  [0 .. pi/2]
// Out: [0 .. 89]
inline int theta_diff_index(double theta_diff)
{
	int tmp = int(theta_diff / (M_PI * 0.5) * BRDF_SAMPLING_RES_THETA_D);
	if (tmp < 0)
		return 0;
	else if (tmp < BRDF_SAMPLING_RES_THETA_D - 1)
		return tmp;
	else
		return BRDF_SAMPLING_RES_THETA_D - 1;
}

inline double theta_diff_rad(int idx){
	return (double)idx * M_PI / 180.0;
}


// Lookup phi_diff index
inline int phi_diff_index(double phi_diff)
{
	// Because of reciprocity, the BRDF is unchanged under
	// phi_diff -> phi_diff + M_PI
	if (phi_diff < 0.0)
		phi_diff += M_PI;

	// In: phi_diff in [0 .. pi]
	// Out: tmp in [0 .. 179]
	int tmp = int(phi_diff / M_PI * BRDF_SAMPLING_RES_PHI_D / 2);
	if (tmp < 0)
		return 0;
	else if (tmp < BRDF_SAMPLING_RES_PHI_D / 2 - 1)
		return tmp;
	else
		return BRDF_SAMPLING_RES_PHI_D / 2 - 1;
}

inline double phi_diff_rad(int idx){
	return (double)idx * M_PI / 180.0;
}


// cross product of two vectors
void cross_product (double* v1, double* v2, double* out)
{
	out[0] = v1[1]*v2[2] - v1[2]*v2[1];
	out[1] = v1[2]*v2[0] - v1[0]*v2[2];
	out[2] = v1[0]*v2[1] - v1[1]*v2[0];
}

// normalize vector
void normalize(double* v)
{
	// normalize
	double len = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
	v[0] = v[0] / len;
	v[1] = v[1] / len;
	v[2] = v[2] / len;
}

// rotate vector along one axis
void rotate_vector(double* vector, double* axis, double angle, double* out)
{
	double temp;
	double cross[3];
	double cos_ang = cos(angle);
	double sin_ang = sin(angle);

	out[0] = vector[0] * cos_ang;
	out[1] = vector[1] * cos_ang;
	out[2] = vector[2] * cos_ang;

	temp = axis[0]*vector[0]+axis[1]*vector[1]+axis[2]*vector[2];
	temp = temp*(1.0-cos_ang);

	out[0] += axis[0] * temp;
	out[1] += axis[1] * temp;
	out[2] += axis[2] * temp;

	cross_product (axis,vector,cross);

	out[0] += cross[0] * sin_ang;
	out[1] += cross[1] * sin_ang;
	out[2] += cross[2] * sin_ang;
}


// convert standard coordinates to half vector/difference vector coordinates
void std_coords_to_half_diff_coords(double theta_in, double fi_in, double theta_out, double fi_out,
								double& theta_half,double& fi_half,double& theta_diff,double& fi_diff )
{

	// compute in vector
	double in_vec_z = cos(theta_in);
	double proj_in_vec = sin(theta_in);
	double in_vec_x = proj_in_vec*cos(fi_in);
	double in_vec_y = proj_in_vec*sin(fi_in);
	double in[3]= {in_vec_x,in_vec_y,in_vec_z};
	normalize(in);


	// compute out vector
	double out_vec_z = cos(theta_out);
	double proj_out_vec = sin(theta_out);
	double out_vec_x = proj_out_vec*cos(fi_out);
	double out_vec_y = proj_out_vec*sin(fi_out);
	double out[3]= {out_vec_x,out_vec_y,out_vec_z};
	normalize(out);


	// compute halfway vector
	double half_x = (in_vec_x + out_vec_x)/2.0f;
	double half_y = (in_vec_y + out_vec_y)/2.0f;
	double half_z = (in_vec_z + out_vec_z)/2.0f;
	double half[3] = {half_x,half_y,half_z};
	normalize(half);

	// compute  theta_half, fi_half
	theta_half = acos(half[2]);
	fi_half = atan2(half[1], half[0]);


	double bi_normal[3] = {0.0, 1.0, 0.0};
	double normal[3] = { 0.0, 0.0, 1.0 };
	double temp[3];
	double diff[3];

	// compute diff vector
	rotate_vector(in, normal , -fi_half, temp);
	rotate_vector(temp, bi_normal, -theta_half, diff);

	// compute  theta_diff, fi_diff
	theta_diff = acos(diff[2]);
	fi_diff = atan2(diff[1], diff[0]);
}


std::vector<double> half_diff_conversion_wrapper(double theta_in, double fi_in, double theta_out, double fi_out){
	double theta_half, fi_half, theta_diff, fi_diff;

	std_coords_to_half_diff_coords(theta_in,fi_in,theta_out,fi_out,theta_half,fi_half,theta_diff,fi_diff);

	return std::vector<double>{theta_half,fi_half,theta_diff,fi_diff};

}



int get_index_from_hall_diff_coord(double theta_half, double theta_diff, double phi_diff){
	int ind = phi_diff_index(phi_diff) +
	theta_diff_index(theta_diff) * BRDF_SAMPLING_RES_PHI_D / 2 +
	theta_half_index(theta_half) * BRDF_SAMPLING_RES_PHI_D / 2 *
						BRDF_SAMPLING_RES_THETA_D;

	//std::cout << theta_half_index(theta_half) << "," << theta_diff_index(theta_diff) << "," << phi_diff_index(phi_diff) << std::endl;

	return ind;
}


std::vector<double> get_half_diff_coord_from_index(int idx){
	int phi_diff_idx,theta_diff_idx,theta_half_idx;

	theta_half_idx = idx / (BRDF_SAMPLING_RES_PHI_D / 2 * BRDF_SAMPLING_RES_THETA_D);
	theta_diff_idx = (idx - theta_half_idx * (BRDF_SAMPLING_RES_PHI_D / 2 * BRDF_SAMPLING_RES_THETA_D)) / (BRDF_SAMPLING_RES_PHI_D / 2);
	phi_diff_idx = (idx - theta_half_idx * (BRDF_SAMPLING_RES_PHI_D / 2 * BRDF_SAMPLING_RES_THETA_D)) % (BRDF_SAMPLING_RES_PHI_D / 2);

	return std::vector<double>{theta_half_rad(theta_half_idx),theta_diff_rad(theta_diff_idx),phi_diff_rad(phi_diff_idx)};
}

std::vector<int> get_half_diff_idxes_from_index(int idx){
	int phi_diff_idx,theta_diff_idx,theta_half_idx;

	theta_half_idx = idx / (BRDF_SAMPLING_RES_PHI_D / 2 * BRDF_SAMPLING_RES_THETA_D);
	theta_diff_idx = (idx - theta_half_idx * (BRDF_SAMPLING_RES_PHI_D / 2 * BRDF_SAMPLING_RES_THETA_D)) / (BRDF_SAMPLING_RES_PHI_D / 2);
	phi_diff_idx = (idx - theta_half_idx * (BRDF_SAMPLING_RES_PHI_D / 2 * BRDF_SAMPLING_RES_THETA_D)) % (BRDF_SAMPLING_RES_PHI_D / 2);

	return std::vector<int>{theta_half_idx,theta_diff_idx,phi_diff_idx};
}


int get_index_from_half_diff_idxes(int theta_half_idx,int theta_diff_idx,int phi_diff_idx){
	return 	phi_diff_idx + theta_diff_idx * BRDF_SAMPLING_RES_PHI_D / 2 +
	theta_half_idx * BRDF_SAMPLING_RES_PHI_D / 2 * BRDF_SAMPLING_RES_THETA_D;
}



PYBIND11_MODULE(merl,m){
	py::class_<MERL_BRDF>(m,"MERL_BRDF")
	.def(py::init<std::string&>())
	.def("look_up",&MERL_BRDF::lookup_wrapper)
	.def("look_up_channel",&MERL_BRDF::lookup_one_channel)
	.def("look_up_hdidx",&MERL_BRDF::lookup_wrapper_hdidxes)
	.def("integrate_second_term_theta_h",&MERL_BRDF::integrate_all_view_theta)
	.def("integrate_second_term_theta_h_monte_carlo",&MERL_BRDF::integrate_all_view_theta_monte_carlo)
	.def_readonly("m_size",&MERL_BRDF::size)
	.def_readwrite("r_channel_unscaled", &MERL_BRDF::vector_data_r)
	.def_readwrite("g_channel_unscaled", &MERL_BRDF::vector_data_g)
	.def_readwrite("b_channel_unscaled", &MERL_BRDF::vector_data_b)
	.def_readonly("r_scale",&MERL_BRDF::RED_SCALE)
	.def_readonly("b_scale",&MERL_BRDF::BLUE_SCALE)
	.def_readonly("g_scale",&MERL_BRDF::GREEN_SCALE);


	m.def("get_index_from_hall_diff_coords", &get_index_from_hall_diff_coord,"params -> [0]:theta_half [1]:theta_diff [2]phi_diff")
	.def("convert_to_hd",&half_diff_conversion_wrapper,"params -> [0]:theta_in [1]:theta_in [2]theta_out [3]phi_out\nreturn list[float] -> [0]:theta_half [1]:phi_half [2]:theta_diff [3]phi_diff")
	.def("reduce_phi_d", &reduce_fi_d)
	.def("get_half_diff_coord_from_index",&get_half_diff_coord_from_index,"return list[float] in rad -> [0]:theta_half [1]:theta_diff [2]phi_diff")
	.def("get_half_diff_idxes_from_index",&get_half_diff_idxes_from_index,"return list[int] in degree/index -> [0]:theta_half [1]:theta_diff [2]phi_diff")
	.def("get_index_from_half_diff_idxes",&get_index_from_half_diff_idxes, "param -> [0]:theta_half [1]:theta_diff [2]phi_diff")
	.def("theta_half_rad",&theta_half_rad)
	.def("theta_diff_rad",&theta_diff_rad)
	.def("phi_diff_rad",&phi_diff_rad);
}