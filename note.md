### Jacobian Weighting in downsampling
The paper suggest that weight the sample using a modified Jacobian so that there is less weight for samples near corners and edges.
The Jacobian is $J(x,y,z) = \frac{1}{(x^2+y^2+z^2)^{\frac{3}{2}}}$. This Jacobian makes sense since if the sample is near corner or edges. The sum of $x^2,y^2,z^2$ would be larger so that the Jacobian is smaller.
However, the code they use the weight of $(x^2+y^2+z^2)^{\frac{3}{2}}$, so the samples near corners and edges have larger weights. This seems contradictory to what the authors state in the paper.

#### About Jacobian
- $ x = 2u $ and $J(u) = \frac{dx}{du} = 2$. This means change of u in one unit causes the change of x in 2 unit
- So for cube mapping to sphere unit direction $r(x,y,z) = \frac{(x,y,z)}{\sqrt{x^2+y^2+z^2}}$
  - $J(x,y,z) = \frac{1}{(x^2 + y^2 + z^2)^{\frac{3}{2}}}$. This means change of cube texture at the center of cube face cause the same amount of change to the sphere surface. But change in the corner/edges part result in less change for sphere surface(there is a compression in volume)
  - In reverse, a change on sphere surface will result in a larger change in cube texture.
    - This is why in the second pass filtering, the code uses a larger miplevel for directions that are near corners and edges.
      - The code add a miplevel of $\frac{1}{2} \log_{2}{(\frac{1}{J(x,y,z)})}$
      - >The sampling function applies a mip-level offset that is a function of the sampling position and adjusts for the Jacobian of the sphere to cube mapping. This offset is simply the contribution of the Jacobian factored out of Equation 4.



### Filtering computation
- Frames
  - Three frames, along x,y,z axis. Think of this as moving Z axis(the UP vector) to either $(1,0,0),(0,1,0)$ or $(0,0,1)$
  - Change of coordinate system for later polynomial fit and parameter computation
    - The texel direction $(x,y,z)$ is considered the new $Z$ axis, so $\vec{Z} = (x,y,z)$. The polar axis of this direction(which face is this dir on) is $\vec{n}$. Then the new x-axis would be $\vec{X} = \vec{n} \times \vec{Z}$. The new Y axis would be $\vec{(x,y,z)} \times \vec{X}$
    - The offset is computed using this new parameterization. $x_i\vec{X} + y_i\vec{Y} + z_i\vec{Z}$
    - Did they do this because some kind of symmetric properties?
- $\theta$ and $\phi$ parameterization for polynomial fit
  - $\theta$ and $\phi$ are more like a modified $u,v$ parameterization where $u,v$ need special handling at $\pm Z$ faces(the up vector)
  - Note: here we use the traditional coordinate system where Z is up, not like in graphics where Z is front/back.
    - For non-up and non-bottom face.(the left right front back) face. Either $x$ or $y$ is $\pm1$. The $\theta$ would be $\pm x$ or $\pm y$ which is non-$\pm 1$(there is $\pm$ because $\theta$ goes counter-clockwise in each face, there are two faces that $x/y$ have the same order as $\theta$, the order of other two faces are reversed)
    - For up/bot face, the $\phi$ is computed as $\pm MAX(x,y)$ and $\theta$ is divided to four parts from the face origin evenly, and the point is _normalized_ to the edge and then the $\theta$ is computed as it is on the four non-up/bot face
- SampleLevel
  - The mipmap level that is generated by the polynomial could be out of bounds. A normal shader will generally clamp this to a valid range. Is this expected???




### Wrong code
```
float calcWeight( float u, float v )
{
	float val = u*u + v*v + 1;
	return val*sqrt( val );
}


float weights[4];
weights[0] = calcWeight( u0, v0 );
weights[1] = calcWeight( u1, v0 );
weights[2] = calcWeight( u0, v1 );
weights[3] = calcWeight( u1, v1 );

const float wsum = 0.5f / ( weights[0] + weights[1] + weights[2] + weights[3] );
[unroll]
for ( int i = 0; i < 4; i++ )
    weights[i] = weights[i] * wsum + .125f;

float3 dir;
float4 color;
switch ( id.z )
{
case 0:
    get_dir_0( dir, u0, v0 );
    color = tex_hi_res.SampleLevel( bilinear, dir, 0 ) * weights[0];

    get_dir_0( dir, u1, v0 );
    color += tex_hi_res.SampleLevel( bilinear, dir, 0 ) * weights[1];

    get_dir_0( dir, u0, v1 );
    color += tex_hi_res.SampleLevel( bilinear, dir, 0 ) * weights[2];

    get_dir_0( dir, u1, v1 );
    color += tex_hi_res.SampleLevel( bilinear, dir, 0 ) * weights[3];
    break;
```


```
// calculate parametrization for polynomial
float Nx = dir[otherAxis0];
float Ny = dir[otherAxis1];
float Nz = adir[axis];
```

### Q
- Downsample with normalization
- What symmetry is used here(same jacobian, same polynomial fit?)
- clip sample level(negative level)?. Sample level Jacobian tuned
- ground truth-how to?
- viewing direction? $cos\theta$
- GGX Kernel?
- $\int L(l)D(h)dl$ -> $\int L(x)B(x)dx$?

- Axis difference?


### Split sum assumption
$n=v=l$
compute groundtruth numerically, loop through all faces, note the Jacobian
importance sampling GGX ndf

### optimization
b(x) pushed back

### Tonemapping
compare pixel value curve?

### synthetic cubemap to test?
$sin$?










### Optimization
For every texel, assume we follow $l=h=v$, we will have an integral of
$\int l(h)D(h)dh = \sum_i^N c_i l_e(x)$ and $l_e(x) = \int l(h)D(h)dh$ 
- Remove l(h), we have $\int D(h)dh = \int \sum_i^N c_i D(h)dh  $ and we project this to cube map
- So we want to optimize $ \int|B(x) - \sum_i^N c_i b_i(x)|dx $ to $0$
 - $B(x)$ is simply the NDF of a GGX material
 - 


Stocastic gradient decent first on one texel to check the target function

impossible to converge if start randomly?(sgd,bfgs,adam) even for simpler constant case
take Jacobian into consideration or not e.g. when pushing back?(currently no)
importance sampling as initial guess? how to convert it to parameters?

use ref table and compare, still high error?




- roughness:
  - 0.00276213059566896
  - 0.007812380793438864
  - 0.02209439000758283
  - 0.062439054105446264
  - 0.17541160386140583
  - 0.4714045207910317
  - 1.0
- 16 roughess
  - 0.005524187436252161
  - 0.013918943967766476
  - 0.03505537971597294
  - 0.08804509063256238
  - 0.21739779351032737
  - 0.48942038369528706
  - 0.816496580927726




- one pushback preparation took 0.0014 seconds to execute.
- one pushing back took 0.0442 seconds to execute.
- computing error took 2.1592 seconds to execute.
- Back propagation took 3.3398 seconds to execute.
- one pushback preparation took 0.0009 seconds to execute.
- one pushing back took 0.0500 seconds to execute.
- computing error took 2.0790 seconds to execute.
- Back propagation took 4.2514 seconds to execute.
- one pushback preparation took 0.0009 seconds to execute.
- one pushing back took 0.0749 seconds to execute.
- computing error took 2.1064 seconds to execute.
- Back propagation took 4.3171 seconds to execute.
- one pushback preparation took 0.0012 seconds to execute.
- one pushing back took 0.0949 seconds to execute.

initialization?

too slow

jacobian level?

shift in filtered image

one texel optimization:
  face 4 u=0.8 v=0.2
  adjust level -> higher value?

use a point, so that it recovers ggx distribution?





TODO:

filter view-dependent

actually render to test?

for each theta? separate?


initialization!!! optimize to importance sampling parameter

change Z-axis of view dependent case

n,l,v relationship. Do we count the part where h is above horizon but l or v are below horizon?(in compute_reference and pushback)

jacobian from h to l/v in reference?

anisotropy of kernel. Horizon clipping?

Change axis, make X/Y match the anisotropic direction? -> The axis should be on the same plane with h,l,v. And it should be perpendicular to l?

Parameterization if there is no frame?
-> using both normal and reflected direction -> lower loss?

change default setting from clipping to not clipping view ndf

To regularize the direction to not go below the hemisphere, some regularization terms should be added


Use VNDF,

Grazing angles and non-grazing angles can not get optimized at the same time
grazing angles brighter?(no sure why, happens both is and filter)

v close to n -> filtered image face 2 u 0.26 0.19. filtered image around that part is wrong