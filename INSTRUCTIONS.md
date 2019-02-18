# CIS 566 Homework 3: Environment Setpiece

## Objective
- Apply your knowledge of noise functions, raymarching, SDFs, lighting,
materials, and post-process effects to make a polished 3D setpiece for
your demo reel.

## Base Code
We have provided the same base code as in homework 2, but __you may choose to
complete this assignment in ShaderToy instead__. If you complete this in
ShaderToy, you must create a ShaderToy account named after your PennKey,
and upload `.txt` files to your Git repository containing the
code used in your ShaderToy implementation, labeling the code if it is
supposed to go in different ShaderToy buffers. __Failure to do so will result in
a score of zero on this assignment.__

## Assignment Requirements
- __(40 points)__ Render a 3D scene using ray marched SDFs.
Your scene should be some sort of setpiece, e.g. a forest
glade, a rustic bedroom, a spaceship cockpit, etc. The general aim for your aesthetic should be "realism" (within reason; we realize you have only just begun to work with SDFs and noise). Your render
should incorporate the following techniques:
  - Animation of the camera or environment elements
  - Three different uses of noise (for color, shape, normal, glow, etc.)
  - Remapping of a value [0, 1] to a set of colors
  - Any of the toolbox functions we have discussed
  - Approximated environmental lighting using three to four directional lights and ambient light
  - SDF-based soft shadows as discussed in [IQ's article on penumbra shadows](http://iquilezles.org/www/articles/rmshadows/rmshadows.htm).

- __(30 points)__ You should also incorporate at least two of the following elements in your scene:
  - Depth of field
  - Ray-based specular reflection
  - Rim lighting
  - Approximated Fresnel reflectivity (more reflectivity at glancing angles)
  - Color remapping
  - Vignette
  - SDF blending
  - Distance fog
  - A raymarched homogeneous medium

- __(20 points)__ Technique mastery: You will be scored on how well you direct the procedural elements in your scene to create a coherent visual. The more natural or deliberate visual effects appear, the higher your score. In other words, if your scene looks like noise functions and SDFs haphazardly combined together, your score will be diminished.

- __(10 points)__ Following the specifications listed
[here](https://github.com/pjcozzi/Articles/blob/master/CIS565/GitHubRepo/README.md),
create your own README.md, renaming the file you are presently reading to
INSTRUCTIONS.md. Don't worry about discussing runtime optimization for this
project. Make sure your README contains the following information:
  - Your name and PennKey
  - Citation of any external resources you found helpful when implementing this
  assignment.
  - A link to your live github.io demo (refer to the pinned Piazza post on
    how to make a live demo through github.io)
  - An explanation of the techniques you used to generate your planet features.
  Please be as detailed as you can; not only will this help you explain your work
  to recruiters, but it helps us understand your project when we grade it!

## Useful Links
- [IQ's Article on Lighting](http://iquilezles.org/www/articles/outdoorslighting/outdoorslighting.htm)


## Submission
Commit and push to Github, then submit a link to your commit on Canvas. Remember to make your own README!

## Inspiration
- [Snail](https://www.shadertoy.com/view/ld3Gz2)
- [Journey Tribute](https://www.shadertoy.com/view/ldlcRf)
- [Stormy Landscape](https://www.shadertoy.com/view/4ts3z2)
- [Volcanic](https://www.shadertoy.com/view/XsX3RB)
- [Elevated](https://www.shadertoy.com/view/MdX3Rr)
- [Rainforest](https://www.shadertoy.com/view/4ttSWf)
- [Canyon](https://www.shadertoy.com/view/MdBGzG)
- [Ladybug](https://www.shadertoy.com/view/4tByz3)
- [Woods](https://www.shadertoy.com/view/XsfGD4)
- [Catacombs](https://www.shadertoy.com/view/lsf3zr)
- [Greek Temple](https://www.shadertoy.com/view/ldScDh)
- [Bridge](https://www.shadertoy.com/view/Mds3z2)
- [Terrain Tubes](https://www.shadertoy.com/view/4sjXzG)

## Extra Credit (20 points maximum)
- __(5 - 20 pts)__ Do some research into more advanced shading techniques such
as ambient occlusion, soft shadows, GGX materials, depth of field, volumetrics,
etc. and implement one of them. The more complex your feature, the more points
you'll earn.
- __(? pts)__ Propose an extra feature of your own!
