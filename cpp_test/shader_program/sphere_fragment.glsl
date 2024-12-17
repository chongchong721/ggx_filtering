#version 430 core
out vec4 FragColor;

in vec3 Normal;
in vec3 Position;

uniform vec3 cameraPos;
uniform samplerCube skybox;
uniform samplerCube filteredMap;
uniform sampler2D secondTermTexture;

/**

For split sum application

We need 1.
- for the first term
    filtered cubemap texture
    world space normal(and then sample the filtered cubemap)
- for the second term
    1D texture of the second term. Given we do not have roughness for MERL material so the table reduced to 1D
    cos_theta_v sample from 1D texture

**/

void main()
{
    // A perfect mirror using a skybox
    vec3 I = normalize(Position - cameraPos); //This is the view direction
    //vec3 R = reflect(I, normalize(Normal));
    float cos_theta = dot(I,Normal); //This is cos(view_theta), which should be used to sample the second sum

    vec4 second_term = texture(secondTermTexture,vec2(0,cos_theta));
    vec4 first_term = texture(filteredMap, Normal);

    FragColor = vec4((first_term * second_term).xyz, 1.0);
}