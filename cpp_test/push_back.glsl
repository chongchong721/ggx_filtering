#version 430 core

#define WHOLE_MIPMAP_OFFSET 6 * (128 * 128 + 64 * 64 + 32 * 32 + 16 * 16 + 8 * 8 + 4 * 4 + 2 * 2)

#define OFFSET_128 128 * 128

const int level_offset[7] = int[7](0, 6*16384, 6*20480, 6*21504, 6*21760, 6*21824, 6*21840);
const int level_edges_offset[7] = int[7](0, 6 * 512, 6 * 768, 6 * 896, 6 * 960, 6 * 992, 6 * 1008);


// The input of this shader is the cubemap mipmap of N * 6 * (128^2 + 64^2 + 32^2 + 16^2 + 8^2 + 4^2 + 2^2)

layout(std430, binding = 1) buffer TextureBuffer{
    float textures[];
};

layout(std430, binding = 3) buffer EdgeMapBuffer{
    int edge_texture_maps[];
};

layout(std430, binding = 4) buffer TmpDebugBuffer{
    float dbg_output[];
};


layout(location = 0) uniform int current_level;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;


vec3 get_xyz(vec2 uv, int face){
    vec3 dir;

    uv = uv * 2.0 - 1.0;

    switch(face){
        case 0:
        dir.x = 1.0f;
        dir.y = uv.y;
        dir.z = -uv.x;
        break;
        case 1:
        dir.x = -1.0f;
        dir.y = uv.y;
        dir.z = uv.x;
        break;
        case 2:
        dir.x = uv.x;
        dir.y = 1.0f;
        dir.z = -uv.y;
        break;
        case 3:
        dir.x = uv.x;
        dir.y = -1.0f;
        dir.z = uv.y;
        break;
        case 4:
        dir.x = uv.x;
        dir.y = uv.y;
        dir.z = 1.0f;
        break;
        case 5:
        dir.x = -uv.x;
        dir.y = uv.y;
        dir.z = -1.0f;
        break;
    }

    return dir;
}



int from_uv_arr_idx_to_texture_idx(ivec2 uv_arr_idx, int face, int level, int res){
    int i = uv_arr_idx.y;
    int j = uv_arr_idx.x;
    int idx = level_offset[level] + res * res * face + i * res + j;
    return idx;
}

/**
* Which idx in the out-of-bound edge texature map should this be in
*/
int from_uv_arr_idx_to_edge_idx(int u, int v, int face,int level, int res){
    int idx = level_edges_offset[level];
    idx = idx + face * res * 4;
    if(u == -1){
        idx = idx + v;
    }else if(u == res){
        idx = idx + res + v;
    }else if(v == -1){
        idx = idx + 2 * res + u;
    }else if(v == res){
        idx = idx + 3 * res + u;
    }else{
        idx = -1;//Sth is wrong
    }
    return idx;
}



//Compute the four bilinear sample location to get the Jacobian
//The uv_arr_idx here must be the exact bilinear sample location(pixel center)
//the return vec4 is u_location_left,u_location_right,v_location_bot,v_location_top

//NOTE: The uv_location we get from this function should be used for the higer-level!
vec4 downsample_uv_pattern(vec2 uv_location ,int level, int res){


    float u_location = uv_location.x;
    float v_location = uv_location.y;


    float inv_res = 1.0 / float(res);

    //How far the bilinear sample is from uv_location;
    float unit = inv_res / 2 * 3 / 4;

    float bilinear_u_left_location = u_location - unit;
    float bilinear_u_right_location = u_location + unit;
    float bilinear_v_bot_location = v_location - unit;
    float bilinear_v_top_location = v_location + unit;

    return vec4(bilinear_u_left_location,bilinear_u_right_location,bilinear_v_bot_location,bilinear_v_top_location);

}


//dir must be on the cube(one coordinate is one)
float jacobian(vec3 dir){
    float sq = dot(dir,dir);
    return 1.0 / pow(sq,1.5);
}


vec4 get_bilinear_jacobian(vec4 uv_location, int face){
    vec4 j;
    vec3 xyz_top_left = get_xyz(vec2(uv_location.x,uv_location.w),face);
    vec3 xyz_top_right = get_xyz(vec2(uv_location.y,uv_location.w),face);
    vec3 xyz_bot_left = get_xyz(vec2(uv_location.x,uv_location.z),face);
    vec3 xyz_bot_right = get_xyz(vec2(uv_location.y,uv_location.z),face);

    j.x = jacobian(xyz_top_left);
    j.y = jacobian(xyz_top_right);
    j.z = jacobian(xyz_bot_left);
    j.w = jacobian(xyz_bot_right);

    return j;
}


float get_sample_uv_location(int idx, float inv_res){
    return float(idx) * inv_res + inv_res / 2.0;
}




void texture_add(ivec2 uv_arr_idx, int face, int level, int res, int dir_idx, float weight){
    /**
    Given a uv(ji) index on a face, find out where in the texture should this be added to

    There are a few different cases:
    // left-out right-out bot-out top-out. top-left-corner top-right-corner bot-left corner bot-right corner and in face case
    **/

    int u_idx = uv_arr_idx.x; int v_idx = uv_arr_idx.y;

    //in boundary
    if(u_idx < res && u_idx >= 0 && v_idx < res && v_idx >=0){
        int tex_idx = from_uv_arr_idx_to_texture_idx(ivec2(u_idx,v_idx),face,level,res);
        tex_idx += dir_idx * WHOLE_MIPMAP_OFFSET;
        textures[tex_idx] += weight;
        return;
    }
    //neighbor_idx is the index itself if this index is within boundary
    int u_neighbor_idx,v_neighbor_idx;
    if(u_idx < 0){
        u_neighbor_idx = 0;
    }else if(u_idx >= res){
        u_neighbor_idx = res - 1;
    }else{
        u_neighbor_idx = u_idx;
    }


    if(v_idx < res){
        v_neighbor_idx = 0;
    }else if(v_idx >= res){
        v_neighbor_idx = res - 1;
    }else{
        v_neighbor_idx = v_idx;
    }


    //corner
    if(
    (u_idx < 0 && v_idx >=res) ||
    (u_idx < 0 && v_idx < 0) ||
    (u_idx >=res && v_idx >=res) ||
    (u_idx >=res && v_idx < 0))
    {
        weight = weight / 3;
        //Two non-corner texel
        int edge_idx1 = from_uv_arr_idx_to_edge_idx(u_neighbor_idx, v_idx,face,level,res);
        int edge_idx2 = from_uv_arr_idx_to_edge_idx(u_idx, v_neighbor_idx,face,level,res);
        int tex_idx1 = edge_texture_maps[edge_idx1];
        int tex_idx2 = edge_texture_maps[edge_idx2];
        tex_idx1 += dir_idx * WHOLE_MIPMAP_OFFSET;
        tex_idx2 += dir_idx * WHOLE_MIPMAP_OFFSET;
        textures[tex_idx1] += weight;
        textures[tex_idx2] += weight;
        int tex_idx = from_uv_arr_idx_to_texture_idx(ivec2(u_neighbor_idx,v_neighbor_idx),face,level,res);
        tex_idx += dir_idx * WHOLE_MIPMAP_OFFSET;
        textures[tex_idx] += weight;


        return;
    }else{
        //edge case
        int edge_idx = from_uv_arr_idx_to_edge_idx(u_idx,v_idx,face,level,res);
        int tex_idx = edge_texture_maps[edge_idx];
        tex_idx += dir_idx * WHOLE_MIPMAP_OFFSET;
        textures[tex_idx] += weight;
        return;
    }

}



void push_back(int current_level, int current_res, int higher_level, int higher_res, int face,int sample_idx ,ivec2 uv_arr_idx, ivec2 uv_idx, vec2 uv_location){
    vec4 bilinear_uv_location = downsample_uv_pattern(uv_location,current_level,current_res);
    vec4 j = get_bilinear_jacobian(bilinear_uv_location,face);

    float j_sum = j.x + j.y + j.z + j.w;

    int higher_res_face_start_idx = WHOLE_MIPMAP_OFFSET * sample_idx + level_offset[higher_level] + face * higher_res * higher_res;
    int current_res_face_start_idx = WHOLE_MIPMAP_OFFSET * sample_idx + level_offset[current_level] + face * current_res * current_res;


    float current_weight = textures[current_res_face_start_idx + uv_arr_idx.y * current_res + uv_arr_idx.x];

    vec4 bilinear_weight;

    bilinear_weight = (0.125f + 0.5 * j / j_sum) * current_weight;
//    bilinear_weight.x = (1/8 + 0.5 * j.x / j_sum) * current_weight;
//    bilinear_weight.y = (1/8 + 0.5 * j.y / j_sum) * current_weight;
//    bilinear_weight.z = (1/8 + 0.5 * j.z / j_sum) * current_weight;
//    bilinear_weight.w = (1/8 + 0.5 * j.w / j_sum) * current_weight;

    float weights[16];
    //create a 16-length array to store all weights (from top->bot then left->right)
    for(int i = 0; i < 4; ++i){
        vec4 portion_row;
        if(i == 0 || i == 3){
            portion_row = vec4(1,3,3,1) / 16.0;
        }else{
            portion_row = vec4(3,9,9,3) / 16.0;
        }
        for(int j = 0; j < 4; ++j){//row
            int tmp_weight_idx = (j / 2) * 2 + i / 2;
            float tmp_weight = bilinear_weight[tmp_weight_idx];
            weights[4*i + j] = tmp_weight * portion_row[j];
            //dbg_output[4*i + j] = weights[4*i+j]
        }
    }


    // We have in total 16 texels in the higher resolution map to add

    //idx i in lower res maps to (2i-1, 2i, 2i+1, 2i+2)^2
    int u = uv_arr_idx.x;
    int v = uv_arr_idx.y;

    ivec4 u_higher_level_idx = ivec4(2*u-1,2*u,2*u+1,2*u+2);
    ivec4 v_higher_level_idx = ivec4(2*v-1,2*v,2*v+1,2*v+2);

    for(int i = 0; i < 4; ++i){
        for (int j = 0; j < 4; ++j){
            texture_add(ivec2(u_higher_level_idx[i],v_higher_level_idx[j]),face,higher_level,higher_res,sample_idx,weights[4*i+j]);
        }
    }
}


void main(){
    //id is the 'ij' index
    int id = int(gl_GlobalInvocationID.x);
    int face_id = int(gl_GlobalInvocationID.y);
    int sample_idx = int(gl_GlobalInvocationID.z);

    int current_res = 2 << (6 - current_level);
    float current_inv_res = 1.0 / float(current_res);


    if(id >= current_res * current_res){
        return;
    }

    dbg_output[0] += 1;

    ivec2 uv_arr_idx;
    //v(i)
    uv_arr_idx.y = id / current_res;
    uv_arr_idx.x = id - uv_arr_idx.y * current_res;

    ivec2 uv_idx;
    uv_idx.x = uv_arr_idx.x;
    uv_idx.y = (current_res - 1) - uv_arr_idx.y;

    vec2 uv_location;
    uv_location.x = get_sample_uv_location(uv_idx.x,current_inv_res);
    uv_location.y = get_sample_uv_location(uv_idx.y,current_inv_res);


    int higher_level = current_level - 1;
    int higher_res = current_res * 2;



    push_back(current_level,current_res,higher_level,higher_res,face_id,sample_idx,uv_arr_idx,uv_idx,uv_location);

}