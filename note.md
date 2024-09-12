### Jacobian Weighting in downsampling
The paper suggest that weight the sample using a modified Jacobian so that there is less weight for samples near corners and edges.
The Jacobian is $J(x,y,z) = \frac{1}{(x^2+y^2+z^2)^{\frac{3}{2}}}$. This Jacobian makes sense since if the sample is near corner or edges. The sum of $x^2,y^2,z^2$ would be larger so that the Jacobian is smaller.
However, the code they use the weight of $(x^2+y^2+z^2)^{\frac{3}{2}}$, so the samples near corners and edges have larger weights. This seems contradictory to what the authors state in the paper.