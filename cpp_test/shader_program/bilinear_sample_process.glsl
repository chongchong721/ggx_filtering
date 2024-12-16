#version 430 core


#define WHOLE_MIPMAP_OFFSET 6 * (128 * 128 + 64 * 64 + 32 * 32 + 16 * 16 + 8 * 8 + 4 * 4 + 2 * 2)

#define OFFSET_128 128 * 128

const int level_offset[7] = int[7](0, 6*16384, 6*20480, 6*21504, 6*21760, 6*21824, 6*21840);
const int level_edges_offset[7] = int[7](0, 6 * 512, 6 * 768, 6 * 896, 6 * 960, 6 * 992, 6 * 1008);

struct BilinearInfo{
  ivec4 arr_idx;
    vec4 uv_location; // vec4[0] u_left [1] u_right [2] v_bot [3] v_top
};

struct SampleInfo {
    vec3 direction;
    float weight;
    float level;
    uint sample_idx;
};

layout(std430, binding = 0) buffer SampleInfoBuffer{
    SampleInfo sampleinfos[];
};

layout(std430, binding = 1) buffer TextureBuffer{
    float textures[];
};

layout(std430, binding = 2) buffer EdgeBuffer{
    float edges[];
};

layout(std430, binding = 3) buffer EdgeMapBuffer{
    int edge_texture_maps[];
};

layout(std430, binding = 4) buffer TmpDebugBuffer{
    float dbg_output[];
};

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;


//Compute the four bilinear sample location to get the Jacobian
//The uv_arr_idx here must be the exact bilinear sample location(pixel center)
//the return vec4 is u_location_left,u_location_right,v_location_bot,v_location_top
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



vec3 get_uv(vec3 direction){
    vec3 dir_copy = direction;

    direction = normalize(direction);

    float x = direction.x;
    float y = direction.y;
    float z = direction.z;

    vec3 abs_direction = abs(direction);

    bool is_x_positive = x > 0.0? true : false;
    bool is_y_positive = y > 0.0? true : false;
    bool is_z_positive = z > 0.0? true : false;

    float max_axis,u,v;

    vec3 ret_info;

    if(is_x_positive && abs_direction.x >= abs_direction.y && abs_direction.x >= abs_direction.z){
        max_axis = abs_direction.x;
        u = -z;
        v = y;
        ret_info.z = 0;
    }
    if(!is_x_positive && abs_direction.x >= abs_direction.y && abs_direction.x >= abs_direction.z){
        max_axis = abs_direction.x;
        u = z;
        v = y;
        ret_info.z = 1;
    }
    if(is_y_positive && abs_direction.y >= abs_direction.x && abs_direction.y >= abs_direction.z){
        max_axis = abs_direction.y;
        u = x;
        v = -z;
        ret_info.z = 2;
    }
    if(!is_y_positive && abs_direction.y >= abs_direction.x && abs_direction.y >= abs_direction.z){
        max_axis = abs_direction.y;
        u = x;
        v = z;
        ret_info.z = 3;
    }
    if(is_z_positive && abs_direction.z >= abs_direction.x && abs_direction.z >= abs_direction.y){
        max_axis = abs_direction.z;
        u = x;
        v = y;
        ret_info.z = 4;
    }
    if(!is_z_positive && abs_direction.z >= abs_direction.x && abs_direction.z >= abs_direction.y){
        max_axis = abs_direction.z;
        u = -x;
        v = y;
        ret_info.z = 5;
    }



    u = 0.5 * (u / max_axis + 1.0f);
    v = 0.5 * (v / max_axis + 1.0f);

//    if(isnan(u)){
//        dbg_output[10] += 1;
//        dbg_output[12] = dir_copy.x;
//        dbg_output[13] = dir_copy.y;
//        dbg_output[14] = dir_copy.z;
//    }
//    if(isnan(v)){
//        dbg_output[11] += 1;
//    }

    ret_info.x = u;
    ret_info.y = v;
    return ret_info;
}



float get_sample_uv_location(int idx, float inv_res){
    return float(idx) * inv_res + inv_res / 2.0;
}


BilinearInfo get_bilinear_info(vec3 uv_info,ivec2 level, int dir_idx){
    //level.x -> level level.y -> level resolution

    //assume 6 faces are stored continuously

    int level_res = level.y;
    int face_idx = int(uv_info.z);

    //now start_idx point to the current face
    float u = uv_info.x;
    float v = uv_info.y;



    float inv_res = 1 / float(level_res);
    int u_idx = int(u / inv_res);
    int v_idx = int(v / inv_res);



    if(u_idx * inv_res + inv_res / 2.0 > u){
        u_idx -= 1;
    }

    if(v_idx * inv_res + inv_res / 2.0 > v){
        v_idx -= 1;
    }

    int u_right = u_idx + 1;
    int v_top = v_idx + 1;


    // Compute bilinear sample location in uv space
    float u_left_location = get_sample_uv_location(u_idx,inv_res);
    float u_right_location = u_left_location + inv_res;
    float v_bot_location = get_sample_uv_location(v_idx,inv_res);
    float v_top_location = v_bot_location + inv_res;

    /**debug code
    */
//    dbg_output[0] = u_left_location;
//    dbg_output[1] = u_right_location;
//    dbg_output[2] = v_bot_location;
//    dbg_output[3] = v_top_location;

    vec4 uv_location = vec4(u_left_location,u_right_location,v_bot_location,v_top_location);

//    dbg_output[0] = float(u_idx);
//    dbg_output[1] = float(u_right);
//    dbg_output[2] = float(v_idx);
//    dbg_output[3] = float(v_top);


    //v is in reverse order compared to array order
    int v_top_array = level_res - 1 - v_top;
    int v_bot_array = level_res - 1 - v_idx;
    int u_left_array = u_idx;
    int u_right_array = u_right;

    // in top-left, top-right, bot-left, bot-right order
//    ivec4 idx = ivec4(
//        v_top_array * level_res + u_left_array,
//        v_top_array * level_res + u_right_array,
//        v_bot_array * level_res + u_left_array,
//        v_bot_array * level_res + u_right_array
//    );
    ivec4 idx = ivec4(
        u_idx,u_right,v_bot_array,v_top_array
    );




    BilinearInfo tmp;
    tmp.arr_idx = idx;
    tmp.uv_location = uv_location;

    return tmp;

}


vec4 process_bilinear_weight(vec2 sample_uv, vec4 bilinear_sample_location, float weight){


    float u_left_location,u_right_location,v_bot_location,v_top_location;
    u_left_location = bilinear_sample_location.x;
    u_right_location = bilinear_sample_location.y;
    v_bot_location = bilinear_sample_location.z;
    v_top_location = bilinear_sample_location.w;

    float u = sample_uv.x;
    float v = sample_uv.y;

    float len = u_right_location - u_left_location;

    float w_top_left, w_top_right, w_bot_left, w_bot_right;

    w_top_left = (u_right_location - u) / len * weight * (v - v_bot_location) / len;
    w_top_right = (u - u_left_location) / len * weight * (v - v_bot_location) / len;
    w_bot_left = (u_right_location - u) / len * weight * (v_top_location - v) / len;
    w_bot_right = (u - u_left_location) / len * weight * (v_top_location - v) / len;

//    if(isnan(w_top_left)){
//        dbg_output[10] += 1;
//        if(dbg_output[10] == 2.0){
//            dbg_output[14] = len;
//            dbg_output[15] = u_right_location;
//            dbg_output[16] = u;
//            dbg_output[17] = u_left_location;
//        }
//
//    }
//    if(isnan(w_top_right)){
//        dbg_output[11] += 1;
//    }
//    if(isnan(w_bot_left)){
//        dbg_output[12] += 1;
//    }
//    if(isnan(w_bot_right)){
//        dbg_output[13] += 1;
//    }

    return vec4(w_top_left,w_top_right,w_bot_left,w_bot_right);

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
        dbg_output[9] = float(tex_idx);
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






void process_bilinear_info(BilinearInfo info, int face, int level ,int res, vec4 weight, int dir_idx){

    int u_left_idx,u_right_idx,v_bot_idx,v_top_idx;
    float u_left_location, u_right_location, v_bot_location, v_top_location;
    u_left_idx = info.arr_idx.x;
    u_right_idx = info.arr_idx.y;
    v_bot_idx = info.arr_idx.z;
    v_top_idx = info.arr_idx.w;

    // The four bilinear sample locations in this level
    u_left_location = info.uv_location.x;
    u_right_location = info.uv_location.y;
    v_bot_location = info.uv_location.z;
    v_top_location = info.uv_location.w;


    float w;

    dbg_output[0] = weight.x;
    dbg_output[1] = weight.y;
    dbg_output[2] = weight.z;
    dbg_output[3] = weight.w;
    dbg_output[4] = float(u_left_idx);
    dbg_output[5] = float(u_right_idx);
    dbg_output[6] = float(v_bot_idx);
    dbg_output[7] = float(v_top_idx);
    dbg_output[8] = float(face);

    //Special case -> boundary
    //top-left
    texture_add(ivec2(u_left_idx,v_top_idx),face,level,res,dir_idx,weight.x);
    texture_add(ivec2(u_right_idx,v_top_idx),face,level,res,dir_idx,weight.y);
    texture_add(ivec2(u_left_idx,v_bot_idx),face,level,res,dir_idx,weight.z);
    texture_add(ivec2(u_right_idx,v_bot_idx),face,level,res,dir_idx,weight.w);
//    w = weight.x;
//    if(u_left_idx < 0){
//        if(v_top_idx < 0){
//            //corner
//            w = w / 3;
//            int edge_idx1 = from_uv_arr_idx_to_edge_idx(u_left_idx, v_bot_idx,face,level,res);
//            int edge_idx2 = from_uv_arr_idx_to_edge_idx(u_right_idx, v_top_idx,face,level,res);
//            int tex_idx1 = edge_texture_maps[edge_idx1];
//            int tex_idx2 = edge_texture_maps[edge_idx2];
//            tex_idx1 += dir_idx * WHOLE_MIPMAP_OFFSET;
//            tex_idx2 += dir_idx * WHOLE_MIPMAP_OFFSET;
//            textures[tex_idx1] += w;
//            textures[tex_idx2] += w;
//
//            int tex_idx = from_uv_arr_idx_to_texture_idx(ivec2(u_right_idx,v_bot_idx),face,level,res);
//            tex_idx += dir_idx * WHOLE_MIPMAP_OFFSET;
//            textures[tex_idx] += w;
//        }else{
//            //left-out
//            int edge_idx = from_uv_arr_idx_to_edge_idx(u_left_idx,v_top_idx,face,level,res);
//            int tex_idx = edge_texture_maps[edge_idx];
//            tex_idx += dir_idx * WHOLE_MIPMAP_OFFSET;
//            textures[tex_idx] += w;
//        }
//    }else{
//        if(v_top_idx < 0){
//            //top-out
//            int edge_idx = from_uv_arr_idx_to_edge_idx(u_left_idx,v_top_idx,face,level,res);
//            int tex_idx = edge_texture_maps[edge_idx];
//            tex_idx += dir_idx * WHOLE_MIPMAP_OFFSET;
//            textures[tex_idx] += w;
//        }else{
//            int tex_idx = from_uv_arr_idx_to_texture_idx(ivec2(u_left_idx,v_top_idx),face,level,res);
//            tex_idx += dir_idx * WHOLE_MIPMAP_OFFSET;
//            textures[tex_idx] += w;
//        }
//    }
//    //top-right
//    w = weight.y;
//    if(u_right_idx >= res){
//        if(v_top_idx < 0){
//            //corner
//            w = w / 3;
//            int edge_idx1 = from_uv_arr_idx_to_edge_idx(u_right_idx, v_bot_idx,face,level,res);
//            int edge_idx2 = from_uv_arr_idx_to_edge_idx(u_left_idx, v_top_idx,face,level,res);
//            int tex_idx1 = edge_texture_maps[edge_idx1];
//            int tex_idx2 = edge_texture_maps[edge_idx2];
//            tex_idx1 += dir_idx * WHOLE_MIPMAP_OFFSET;
//            tex_idx2 += dir_idx * WHOLE_MIPMAP_OFFSET;
//            textures[tex_idx1] += w;
//            textures[tex_idx2] += w;
//
//            int tex_idx = from_uv_arr_idx_to_texture_idx(ivec2(u_left_idx,v_bot_idx),face,level,res);
//            tex_idx += dir_idx * WHOLE_MIPMAP_OFFSET;
//            textures[tex_idx] += w;
//        }else{
//            //right-out
//            int edge_idx = from_uv_arr_idx_to_edge_idx(u_right_idx,v_top_idx,face,level,res);
//            int tex_idx = edge_texture_maps[edge_idx];
//            tex_idx += dir_idx * WHOLE_MIPMAP_OFFSET;
//            textures[tex_idx] += w;
//        }
//    }else{
//        if(v_top_idx < 0){
//            //top-out
//            int edge_idx = from_uv_arr_idx_to_edge_idx(u_right_idx,v_top_idx,face,level,res);
//            int tex_idx = edge_texture_maps[edge_idx];
//            tex_idx += dir_idx * WHOLE_MIPMAP_OFFSET;
//            textures[tex_idx] += w;
//        }else{
//            int tex_idx = from_uv_arr_idx_to_texture_idx(ivec2(u_right_idx,v_top_idx),face,level,res);
//            tex_idx += dir_idx * WHOLE_MIPMAP_OFFSET;
//            textures[tex_idx] += w;
//        }
//    }
//    //bot-left
//    w = weight.z;
//    if(u_left_idx < 0){
//        if(v_bot_idx >= res){
//            //corner
//            w = w / 3;
//            int edge_idx1 = from_uv_arr_idx_to_edge_idx(u_left_idx, v_top_idx,face,level,res);
//            int edge_idx2 = from_uv_arr_idx_to_edge_idx(u_right_idx, v_bot_idx,face,level,res);
//            int tex_idx1 = edge_texture_maps[edge_idx1];
//            int tex_idx2 = edge_texture_maps[edge_idx2];
//            tex_idx1 += dir_idx * WHOLE_MIPMAP_OFFSET;
//            tex_idx2 += dir_idx * WHOLE_MIPMAP_OFFSET;
//            textures[tex_idx1] += w;
//            textures[tex_idx2] += w;
//
//            int tex_idx = from_uv_arr_idx_to_texture_idx(ivec2(u_right_idx,v_top_idx),face,level,res);
//            tex_idx += dir_idx * WHOLE_MIPMAP_OFFSET;
//            textures[tex_idx] += w;
//        }else{
//            //left-out
//            int edge_idx = from_uv_arr_idx_to_edge_idx(u_left_idx,v_bot_idx,face,level,res);
//            int tex_idx = edge_texture_maps[edge_idx];
//            tex_idx += dir_idx * WHOLE_MIPMAP_OFFSET;
//            textures[tex_idx] += w;
//        }
//    }else{
//        if(v_bot_idx >= res){
//            //bot-out
//            int edge_idx = from_uv_arr_idx_to_edge_idx(u_left_idx,v_bot_idx,face,level,res);
//            int tex_idx = edge_texture_maps[edge_idx];
//            tex_idx += dir_idx * WHOLE_MIPMAP_OFFSET;
//            textures[tex_idx] += w;
//        }else{
//            int tex_idx = from_uv_arr_idx_to_texture_idx(ivec2(u_left_idx,v_bot_idx),face,level,res);
//            tex_idx += dir_idx * WHOLE_MIPMAP_OFFSET;
//            textures[tex_idx] += w;
//        }
//    }
//    //bot-right
//    w = weight.w;
//    if(u_right_idx >= res){
//        if(v_bot_idx >= res){
//            //corner
//            w = w / 3;
//            int edge_idx1 = from_uv_arr_idx_to_edge_idx(u_right_idx, v_top_idx,face,level,res);
//            int edge_idx2 = from_uv_arr_idx_to_edge_idx(u_left_idx, v_bot_idx,face,level,res);
//            int tex_idx1 = edge_texture_maps[edge_idx1];
//            int tex_idx2 = edge_texture_maps[edge_idx2];
//            tex_idx1 += dir_idx * WHOLE_MIPMAP_OFFSET;
//            tex_idx2 += dir_idx * WHOLE_MIPMAP_OFFSET;
//            textures[tex_idx1] += w;
//            textures[tex_idx2] += w;
//
//            int tex_idx = from_uv_arr_idx_to_texture_idx(ivec2(u_left_idx,v_top_idx),face,level,res);
//            tex_idx += dir_idx * WHOLE_MIPMAP_OFFSET;
//            textures[tex_idx] += w;
//        }else{
//            //right-out
//            int edge_idx = from_uv_arr_idx_to_edge_idx(u_right_idx,v_bot_idx,face,level,res);
//            int tex_idx = edge_texture_maps[edge_idx];
//            tex_idx += dir_idx * WHOLE_MIPMAP_OFFSET;
//            textures[tex_idx] += w;
//        }
//    }else{
//        if(u_right_idx >= res){
//            //bot-out
//            int edge_idx = from_uv_arr_idx_to_edge_idx(u_right_idx,v_bot_idx,face,level,res);
//            int tex_idx = edge_texture_maps[edge_idx];
//            tex_idx += dir_idx * WHOLE_MIPMAP_OFFSET;
//            textures[tex_idx] += w;
//        }else{
//            int tex_idx = from_uv_arr_idx_to_texture_idx(ivec2(u_right_idx,v_bot_idx),face,level,res);
//            tex_idx += dir_idx * WHOLE_MIPMAP_OFFSET;
//            textures[tex_idx] += w;
//        }
//    }

}


void main(){

    int id = int(gl_GlobalInvocationID.x);

    if(id >= 1){
        return;
    }

    int sample_idx = int(gl_GlobalInvocationID.y);



    int info_idx = sample_idx * 24 + id;



    SampleInfo info =  sampleinfos[info_idx];

//    dbg_output[0] = info.direction.x;
//    dbg_output[1] = info.direction.y;
//    dbg_output[2] = info.direction.z;


    vec3 uv_info = get_uv(info.direction);

//    dbg_output[0] = float(uv_info.x);
//    dbg_output[1] = float(uv_info.y);
//    dbg_output[2] = uv_info.z;


    int face = int(uv_info.z);

    int higher_level = int(floor(info.level));
    int lower_level = int(ceil(info.level));

    int higher_res = 2 << (6 - higher_level);
    int lower_res = 2 << (6 - lower_level);

    float higher_distance = info.level - float(higher_level);

    float higher_weight = (1.0 - higher_distance) * info.weight;
    float lower_weight = info.weight - higher_weight;


    BilinearInfo lower_level_info = get_bilinear_info(uv_info,ivec2(lower_level,lower_res),sample_idx);
    BilinearInfo higher_level_info = get_bilinear_info(uv_info,ivec2(higher_level,higher_res),sample_idx);


    vec4 lower_weight_bilinear = process_bilinear_weight(uv_info.xy,lower_level_info.uv_location,lower_weight);
    vec4 higher_weight_bilinear = process_bilinear_weight(uv_info.xy,higher_level_info.uv_location,higher_weight);

    process_bilinear_info(lower_level_info,face,lower_level,lower_res,lower_weight_bilinear,sample_idx);
    process_bilinear_info(higher_level_info,face,higher_level,higher_res,higher_weight_bilinear,sample_idx);

}