#version 430
out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube skybox;

void main()
{
    vec3 hdrColor = texture(skybox, TexCoords).rgb;

    // Reinhard tone mapping
    vec3 toneMapped = hdrColor / (hdrColor + vec3(1.0));

    // Gamma correction
    toneMapped = pow(toneMapped, vec3(1.0/2.2));

    FragColor = vec4(toneMapped, 1.0);
}