#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;

in vec2 fs_Pos;
out vec4 out_Col;


/* --------------------------------------------------------------------------*
 *   Constants
 * --------------------------------------------------------------------------*/

const float FOV = 45.0;
const float EPSILON = 0.001;
const float FAR_CLIP = 1000.0;
const int MAX_ITERATIONS = 2048;

// If this is true, normals will be less accurate, but cheaper
const bool CHEAP_NORMALS = false;

const vec3 NULLV = vec3(0);
const vec3 BLACK = NULLV;
const vec3 RED = vec3(1, 0, 0);
const vec3 GREEN = vec3(0, 1, 0);
const vec3 BLUE = vec3(0, 0, 1);

const int MATERIAL_BG = -1;
const int MATERIAL_LAMBERT = 0;
const int MATERIAL_BLINNPHONG = 1;
const int MATERIAL_GLOSSY = 2; // It's basically blinn-phong, with glossy reflections
const int MATERIAL_REFRACTIVE = 3; // Transparent. Refracts whatever's behind it.

const float AMBIENT_LIGHT = 0.15;
const int LIGHT_DIRECTIONAL = 0;
const int LIGHT_POINT = 1;
const int NUM_LIGHTS = 2;

const vec3 SEED3 = vec3(0.31415, 0.6456, 0.23432);

/* --------------------------------------------------------------------------*
 *   Structs
 * --------------------------------------------------------------------------*/

/*
 * Contains most of the nonglobal data used in material computations.
 * Other data such as fragment position and normal are stored in the
 * Surface struct.
 */
struct Material {
    int type;           // Type of shading used. See constants above
    vec3 baseColor;     // Base color used in most materials
    float shininess;    // The blinn-phong specularity exponent
    float reflexivity;  // The degree to which reflexive materials reflect (in [0, 1])
    float ior;          // The index of refraction used for refraction
    float attenuation;  // How quickly light fades in a transparent object. 0 = not at all
};

/*
 * All the data pertaining to an SDF check.
 * Material and base color are meaningless if distance > EPSILON
 */
struct SDFData {
    float distance;
    Material material;
};

/*
 * All the data pertaining to a point on an SDF generated surface.
 * If the ray doesn't hit a surface, it also represents the
 * background fragment hit by the cast ray.
 */
struct Surface {
    vec3 position;
    vec3 normal;
    vec3 hitRay;        // The direction of the ray that hit this point;
    Material material;
    int maxIterations;  // The maximum number of iterations used to get this surface point
};

struct Light {
    int type;
    float intensity;
    vec3 position;
    vec3 direction;
    vec3 color;
};


/* --------------------------------------------------------------------------*
 *   Utility Functions
 * --------------------------------------------------------------------------*/

float random1(vec3 p , vec3 seed) {
    return fract(sin(dot(p + seed, vec3(987.654, 123.456, 531.975))) * 85734.3545);
}


 Light getLight(int index) {
     index = clamp(index, 0, NUM_LIGHTS);
     if (index == 0) {
         return Light(LIGHT_DIRECTIONAL, 1.0, NULLV, normalize(vec3(1, -1, 0)), vec3(1, 1, 1));
     }
     if (index == 1) {
         return Light(LIGHT_DIRECTIONAL, 1.0, NULLV, normalize(vec3(-1, -1, 0)), vec3(1, 1, 1));
     }
 }

float dot2(vec3 v) {
    return dot(v, v);
}

vec3 getCheckerPattern(vec3 p) {
    ivec3 grid = ivec3(floor(p));
    if ((grid.x + grid.y + grid.z) % 2 == 0) {
        return vec3(0.3);
    }
    else {
        return vec3(0.7);
    }
}

float vmax(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

float smoothInOut(float t) {
    float a = (cos(2.0 * 3.14159 * (t + 0.5)) + 1.0) * 0.5;
    return a;
}

float doubleSmoothstep(float min, float mid, float max, float t) {
    t = clamp(t, 0.0, 1.0);
    return t < 0.5 ? 
        mix(min, mid, smoothstep(0.0, 1.0, t * 2.0)):
        mix(mid, max, smoothstep(0.0, 1.0, (t - 0.5) * 2.0));
}

bool vLess(vec3 v1, vec3 v2) {
    return v1.x < v2.x && v1.y < v2.y && v1.z < v2.z;
}

bool vGreater(vec3 v1, vec3 v2) {
    return vLess(v2, v1);
}

bool vLess(vec2 v1, vec2 v2) {
    return v1.x < v2.x && v1.y < v2.y;
}

bool vGreater(vec2 v1, vec2 v2) {
    return vLess(v2, v1);
}

bool withinRect(vec2 point, vec2 rectCenter, vec2 rectRadii) {
    return vLess(abs(point - rectCenter), rectRadii);
}

bool withinBox(vec3 point, vec3 boxCenter, vec3 boxRadii) {
    return vLess(abs(point - boxCenter), boxRadii);
}

/* --------------------------------------------------------------------------*
 *   Bounding Box Computation
 * --------------------------------------------------------------------------*/

/*
 * Checks if a ray, given its origin and direction, would intersect with a box, given
 * it's bottom corner and dimensions. Returns a vec4, where the w coordinate is 1 if
 * the ray intersects, and 0 otherwise. The xyz coordinates specify where the ray intersects
 * the bounding box, if it does at all.
 */

bool intersectBoundingBox(vec3 rayOrigin, vec3 rayDir, vec3 boxCenter, vec3 boxRadii) {

    vec3 boxCorner = boxCenter - boxRadii;
    vec3 boxDimensions = boxRadii * 2.0;
    vec3 topCorner = boxCorner + boxDimensions;

    if (withinBox(rayOrigin, boxCenter, boxRadii)) {
        return true;
    }

    if (abs(rayDir.x) > EPSILON) {
        float txMin = min((boxCorner.x - rayOrigin.x) / rayDir.x, (topCorner.x  - rayOrigin.x) / rayDir.x);
        vec3 testX = rayOrigin + txMin * rayDir;
        if (txMin >= 0.0 && withinRect(testX.yz, boxCenter.yz, boxRadii.yz)) {
            return true;
        }
    }

    if (abs(rayDir.y) > EPSILON) {
        float tyMin = min((boxCorner.y - rayOrigin.y) / rayDir.y, (topCorner.y  - rayOrigin.y) / rayDir.y);
        vec3 testY = rayOrigin + tyMin * rayDir;
        if (tyMin >= 0.0 && withinRect(testY.xz, boxCenter.xz, boxRadii.xz)) {
            return true;
        }
    }

    if (abs(rayDir.z) > EPSILON) {
        float tzMin = min((boxCorner.z - rayOrigin.z) / rayDir.z, (topCorner.z  - rayOrigin.z) / rayDir.z);
        vec3 testZ = rayOrigin + tzMin * rayDir;
        if (withinRect(testZ.xy, boxCenter.xy, boxRadii.xy)) {
            return true;
        }
    }

    return false;
}

/* --------------------------------------------------------------------------*
 *   Spacial transformations
 * --------------------------------------------------------------------------*/

vec3 twistSpace(vec3 p, float angle) {
    float c = cos(angle * (p.y + 1.0));
    float s = sin(angle * (p.y + 1.0));
    mat2 rotation = mat2(c, -s, s, c);
    return vec3(rotation * p.xz, p.y);
}


/* --------------------------------------------------------------------------*
 *   SDF formulas
 * --------------------------------------------------------------------------*/

// Sphere
float sphereSdf(vec3 p, vec3 offset, float radius) {
    return distance(p, offset) - radius;
}

// Box
float boxSdf(vec3 p, vec3 boxCenter, vec3 boxDimensions) {
    return vmax(abs(p - boxCenter) - boxDimensions);
}

// Quad
float quadSdf(vec3 p, vec3 offset, float w, float l) {
    return boxSdf(p, offset, vec3(l / 2.0, 0, w / 2.0));
}

// Cylinder
float cylinderSdf(vec3 p, vec3 center, float height, float radius) {
    vec3 pos = p - center;
    vec2 d = abs(vec2(length(pos.xz), pos.y)) - vec2(radius, height / 2.0);
    return min(max(d.x, d.y),0.0) + length(max(d, 0.0));
}

// Cone
float coneSdf(vec3 p, vec3 center, float h, float r1, float r2) {
    vec3 pos = p - center;
    vec2 q = vec2( length(pos.xz), pos.y );
    
    vec2 k1 = vec2(r2,h);
    vec2 k2 = vec2(r2-r1,2.0*h);
    vec2 ca = vec2(q.x-min(q.x,(q.y < 0.0)?r1:r2), abs(q.y)-h);
    vec2 cb = q - k1 + k2*clamp( dot(k1-q,k2)/dot(k2, k2), 0.0, 1.0 );
    float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return s*sqrt( min(dot(ca, ca),dot(cb, cb)) );
}

// Torus
float torusSdf(vec3 p, vec3 torusCenter, float majorRadius, float minorRadius) {
    vec3 pos = p - torusCenter;
    return length(vec2(length(pos.xz) - majorRadius, pos.y)) - minorRadius;
}

// Shape Blend
float blendSdf(float d1, float d2, float t) {
    return mix(d1, d2, t);
}

// Union
float unionSdf(float d1, float d2, float smoothness) {
    smoothness = smoothness > 0.0 ? max(smoothness, 0.00001) : min(smoothness, 0.00001);
    float t = clamp(0.5 + 0.5 * (-d1 + d2) / smoothness, 0.0, 1.0);
    return mix(d2, d1, t) - smoothness * t * (1.0 - t);
}
float unionSdf(float d1, float d2) {
    return min(d1, d2);
}
float union3Sdf(float d1, float d2, float d3) {
    return min(d1, min(d2, d3));
}

// Difference
float subtractSdf(float d1, float d2, float smoothness) {
    return unionSdf(d1, -d2, -smoothness);
}
float subtractSdf(float d1, float d2) {
    return max(-d1, d2);
}

// Intersection
float intersectSdf(float d1, float d2, float smoothness) {
    return unionSdf(d1, d2, -smoothness);
}
float intersectSdf(float d1, float d2) {
    return max(d1, d2);
}

/*
 * Used to add another SDF to the scene (within totalSdf).
 * Note that unlike unionSdf, this function does not smooth SDFs,
 * and the SDFs added can have different materials
 */
SDFData addSdf(SDFData sdf1, float sdf2, Material material) {
    if (sdf1.distance > sdf2) {
        return SDFData(sdf2, material);
    }
    else {
        return sdf1;
    }
}

SDFData addSdf(float sdf, Material material) {
    return SDFData(sdf, material);
}

/*                    *----------*
 *                    | *------* |
 *                    | |      | |
 *                    | |      | |
 *                    | |      | |
 *                    | |      | |
 *                    | |      | |
 *                    | |      | |
 *                    | |      | |
 *                    | |      | |
 *                *---* |      | *---*
 *                 \ *--*      *--* /
 *                  \ \          / /
 *                   \ \        / /
 *                    \ \      / /
 *                     \ \    / /
 *                      \ \  / /
 *                       \ \/ /
 *                        \  /
 *                         \/
 *
 *
 * I was just tired of scrolling so much to find this function
 * The accumulaton of every SDF in the scene.
 * SDFs should be included through the addSdf function, along with a material
 */
SDFData totalSdf(vec3 p, vec3 origin, vec3 rayDir, bool ignoreBoxes) {
    Material checker = Material(MATERIAL_LAMBERT, getCheckerPattern(p), 16.0, 0.3, 1.0, 1.0);
    SDFData data = addSdf(quadSdf(p, vec3(0, -2, 0), 10.0, 10.0), checker);
    float seconds = u_Time / 60.0;

    // --- Hourglass -----------------------
    Material glassMat = Material(MATERIAL_REFRACTIVE, vec3(0, 0.5, 0), 20.0, 1.0, 1.2, 1.0);
    Material standMat = Material(MATERIAL_BLINNPHONG, vec3(0.3, 0.15, 0.02), 16.0, 0.1, 1.0, 1.0);
    vec3 sandColor1 = vec3(0.9, 0.6, 0.4) + vec3(random1(floor(p * 100.0), SEED3) - 1.0) * 0.2;
    Material sandMat1 = Material(MATERIAL_LAMBERT, sandColor1, 0.0, 0.0, 0.0, 0.0);

    float glass = FAR_CLIP * 2.0;
    float stand = FAR_CLIP * 2.0;
    float topSand = FAR_CLIP * 2.0;
    float bottomSand = FAR_CLIP * 2.0;

    if (intersectBoundingBox(origin, rayDir, vec3(0, 1.5, 0), vec3(2.65, 2.9, 2.65)) || ignoreBoxes) {
        float glassHull = unionSdf(
            unionSdf(
                coneSdf(p, vec3(0, 0.8, 0), 1.0, 1.25, 0.0),
                intersectSdf(
                    sphereSdf(p, vec3(0, -0.5, 0), 1.5),
                    boxSdf(p, vec3(0, 0.25, 0), vec3(1.5, 0.75, 1.5))
                ),
                0.3
            ),
            unionSdf(
                coneSdf(p, vec3(0, 2.2, 0), 1.0, 0.00, 1.25),
                intersectSdf(
                    sphereSdf(p, vec3(0, 3.5, 0), 1.5),
                    boxSdf(p, vec3(0, 2.75, 0), vec3(1.5, 0.75, 1.5))
                ),
                0.3
            ),
            0.2
        );

        glass = abs(glassHull) - 0.075;

        float twist1 = unionSdf(
            cylinderSdf(p, vec3(-0.15, 1.5, 2.0), 5.0, 0.15),
            cylinderSdf(p, vec3(0.15, 1.5, 2.0), 5.0, 0.15),
            0.03
        );

        float twist2 = unionSdf(
            cylinderSdf(p, vec3(0.075 + sqrt(3.0), 1.5, 0.075 * sqrt(3.0) - 1.0), 5.0, 0.15),
            cylinderSdf(p, vec3(-0.075 + sqrt(3.0), 1.5, -0.075 * sqrt(3.0) - 1.0), 5.0, 0.15),
            0.03
        );

        float twist3 = unionSdf(
            cylinderSdf(p, vec3(-0.075 - sqrt(3.0), 1.5, 0.075 * sqrt(3.0) - 1.0), 5.0, 0.15),
            cylinderSdf(p, vec3(0.075 - sqrt(3.0), 1.5, -0.075 * sqrt(3.0) - 1.0), 5.0, 0.15),
            0.03
        );

        stand =
        unionSdf(
            unionSdf(
                unionSdf(
                    coneSdf(p, vec3(0, -0.75, 0), 0.2, 2.6, 2.4),
                    cylinderSdf(p, vec3(0, -1.15, 0), 0.4, 2.6)
                ),
                unionSdf(
                    coneSdf(p, vec3(0, 3.75, 0), 0.2, 2.4, 2.6),
                    cylinderSdf(p, vec3(0, 4.15, 0), 0.4, 2.6)
                )
            ),
            union3Sdf(twist1, twist2, twist3),
            0.1
        );

        float totalTime = 100.0;
        float t = clamp(seconds / totalTime, 0.0, 1.0);
        float bottomSandOffset = mix(-1.23, 0.0, sqrt(t));
        float topSandoffset = mix(0.0, 0.90, t * t);
        bottomSand = intersectSdf(
            unionSdf(
                sphereSdf(p, vec3(0, -1.3 + bottomSandOffset, 0), 2.0),
                sphereSdf(p, vec3(0, 0.5 + bottomSandOffset, 0), 0.4),
                0.3
            ),
            glassHull + 0.13,
            0.1
        );
        topSand = intersectSdf(
            subtractSdf(
                sphereSdf(p, vec3(0, 3.25, 0.0), 1.8),
                sphereSdf(p, vec3(0, 4.2 - topSandoffset, 0.0), 1.7),
                0.3
            ),
            glassHull + 0.13,
            0.1
        );

        vec3 sandColor2 = vec3(0.9, 0.6, 0.4) + vec3(random1(floor((p + vec3(0, 2.0 * topSandoffset, 0)) * 100.0), SEED3) - 1.0) * 0.2;
        Material sandMat2 = Material(MATERIAL_LAMBERT, sandColor2, 0.0, 0.0, 0.0, 0.0);

        data = addSdf(data, stand, standMat);
        data = addSdf(data, bottomSand, sandMat1);
        data = addSdf(data, topSand, sandMat2);
        data = addSdf(data, glass, glassMat);
    }

    /*
    float sand =
    blendSdf(
        coneSdf(p, vec3(0, 0.8, 0), 0.00, 0.5, 0.5),
        unionSdf(
            coneSdf(p, vec3(0, 2, 0), 0.05, 1.0, 1.0) - 0.3,
            sphereSdf(p, vec3(0, 2.5, 0), 0.3),
            0.9
        ),
        smoothstep(0.0, 1.0, clamp(u_Time * 0.001, 0.0, 1.0))
    );*/

    return data;
}

/*
 * Examines the entire SDF map and uses gradients to find what the
 * surface normal would be at a point in 3D space. If the point isn't
 * on a surface, the computation won't be very useful.
 */
vec3 getNormal(vec3 p) {
    vec3 dx = vec3(EPSILON, 0, 0);
    vec3 dy = vec3(0, EPSILON, 0);
    vec3 dz = vec3(0, 0, EPSILON);

    float gradX = 0.0;
    float gradY = 0.0;
    float gradZ = 0.0;

    if (CHEAP_NORMALS) {
        float pointDist = (totalSdf(p, NULLV, NULLV, true)).distance;
        gradX = totalSdf(p + dx, NULLV, NULLV, true).distance - pointDist;
        gradY = totalSdf(p + dy, NULLV, NULLV, true).distance - pointDist;
        gradZ = totalSdf(p + dz, NULLV, NULLV, true).distance - pointDist;
    }
    else {
        gradX = totalSdf(p + dx, NULLV, NULLV, true).distance - totalSdf(p - dx, NULLV, NULLV, true).distance;
        gradY = totalSdf(p + dy, NULLV, NULLV, true).distance - totalSdf(p - dy, NULLV, NULLV, true).distance;
        gradZ = totalSdf(p + dz, NULLV, NULLV, true).distance - totalSdf(p - dz, NULLV, NULLV, true).distance;
    }
    return -normalize(vec3(gradX, gradY, gradZ));
}


/*
 * Perform the actual raycasting. This does not have to be done from
 * the camera, but except for reflections, etc., it usually is
 */
Surface raycast(vec3 origin, vec3 rayDir, int maxIterations) {
    int curIteration = 0;
    vec3 curPoint = origin + rayDir * 2.0 * EPSILON;
    Material nullMaterial = Material(-1, vec3(0), 0.0, 0.0, 0.0, 0.0);
    Surface surface = Surface(curPoint, NULLV, rayDir, nullMaterial, maxIterations);
    SDFData sdf;

    while (curIteration < maxIterations) {
        curIteration++;
        sdf = totalSdf(curPoint, origin, rayDir, false);
        float distance = abs(sdf.distance);

        // Set the cast data properities to correspond with the surface
        if (distance < EPSILON) {
            surface.position = curPoint;
            surface.normal = getNormal(curPoint);
            surface.material = sdf.material;
            return surface;
        }

        if (distance > FAR_CLIP) {
            break;
        }
        curPoint = curPoint + rayDir * distance;
    }

    // Set the cast data properities to correspond with the background.
    surface.material = Material(MATERIAL_BG, NULLV, 0.0, 0.0, 0.0, 0.0);
    surface.position = curPoint; // Set for debugging purposes;
    return surface;
}


/* --------------------------------------------------------------------------*
 *   Material Calculations
 * --------------------------------------------------------------------------*/

 // Because GLSL doesn't allow for recursion, we have to redeclare a few functions
 // to fake it.
vec3 calculateColorRecursive(Surface surface, vec3 cameraForward);
vec3 calculateColorRecursive2(Surface surface, vec3 cameraForward);

vec3 calculateBackground(Surface surface, vec3 cameraForward) {
    vec3 hitRay = surface.hitRay;
    vec3 bgColor = vec3(abs(normalize(hitRay)));
    return bgColor;
}

vec3 calculateLambert(Surface surface) {
    Material mat = surface.material;
    vec3 lambertColor = vec3(0);
    vec3 color = vec3(0);
    for (int i = 0; i < NUM_LIGHTS; i++) {
        Light light = getLight(i);
        if (light.type == 0) {
            float lambFactor = clamp(dot(light.direction * light.intensity, surface.normal), AMBIENT_LIGHT, light.intensity);
            lambertColor += lambFactor * mat.baseColor * light.color;
        }
    }

    color = lambertColor;
    return clamp(color, vec3(0), vec3(1));
}

vec3 calculateBlinnPhong(Surface surface, vec3 cameraForward) {
    Material mat = surface.material;
    float shininess = mat.shininess;
    vec3 lambertColor = vec3(0);
    vec3 blinnphongColor = vec3(0);
    vec3 color = vec3(0);

    for (int i = 0; i < NUM_LIGHTS; i++) {
        Light light = getLight(i);
        if (light.type == LIGHT_DIRECTIONAL) {
            float lambFactor = clamp(dot(light.direction * light.intensity, surface.normal), AMBIENT_LIGHT, light.intensity);
            lambertColor += lambFactor * mat.baseColor * light.color;

            vec3 halfVec = normalize(light.direction + cameraForward);
            float specAngle = max(dot(surface.normal, halfVec), 0.0);
            float spec = pow(specAngle, shininess) * light.intensity;
            blinnphongColor += light.color * spec;
        }
    }

    color = lambertColor + blinnphongColor;
    return clamp(color, vec3(0), vec3(1));
}

// Turns out I don't need to calculate the reflection vector manually. Imagine that.
vec3 calculateGlossy(Surface surface, vec3 cameraForward) {
    Material mat = surface.material;
    float shininess = mat.shininess;
    float reflexivity = mat.reflexivity;
    vec3 lambertColor = vec3(0);
    vec3 blinnphongColor = vec3(0);
    vec3 glossyColor = vec3(0);

    for (int i = 0; i < NUM_LIGHTS; i++) {
        Light light = getLight(i);
        if (light.type == LIGHT_DIRECTIONAL) {
            float lambFactor = clamp(dot(light.direction * light.intensity, surface.normal), AMBIENT_LIGHT, light.intensity);
            lambertColor += lambFactor * mat.baseColor * light.color;

            vec3 halfVec = normalize(light.direction + cameraForward);
            float specAngle = max(dot(surface.normal, halfVec), 0.0);
            float spec = pow(specAngle, shininess) * light.intensity;
            blinnphongColor += light.color * spec;
        }
    }

    vec3 reflectedRay = reflect(surface.hitRay, surface.normal);

    Surface reflectedSurface = raycast(surface.position, reflectedRay, int(float(surface.maxIterations) * 0.5));
    glossyColor = calculateColorRecursive(reflectedSurface, cameraForward);
    float fresnel = clamp(1.0 - dot(surface.normal, cameraForward), 0.0, 1.0);
    float reflectionCoefficient = doubleSmoothstep(0.0, fresnel, 1.0, mat.reflexivity);

    vec3 color = 
        (AMBIENT_LIGHT * mat.baseColor + lambertColor) * (1.0 - reflectionCoefficient) + 
        glossyColor * reflectionCoefficient +
        blinnphongColor;
    return clamp(color, vec3(0), vec3(1));
}

// I literally spent days working on this, and then I learned that GLSL has a built in
// function that automatically calculates refraction vectors, so that's how my week's been
// going.
vec3 calculateRefractive(Surface surface, vec3 cameraForward) {
    vec3 color;
    Material mat = surface.material;
    vec3 light = vec3(0, -1, 0);
    float lambertianFactor = dot(light, surface.normal);
    vec3 lambertColor = clamp(lambertianFactor * mat.baseColor, vec3(0), vec3(1));

    vec3 halfVec = normalize(light + cameraForward);
    float shininess = mat.shininess;
    float specAngle = max(dot(surface.normal, halfVec), 0.0);
    float spec = pow(specAngle, shininess) * 0.8;
    vec3 bpColor = vec3(1) * spec;

    // Calculate the direction of the once refracted ray
    vec3 refractColor;
    vec3 refractedRay = refract(surface.hitRay, -surface.normal, 1.0 / mat.ior);

    // Raycast to see where the refracted ray hits itself
    vec3 offsetOrigin1 = surface.position + refractedRay * 50.0 * EPSILON;
    Surface selfSurface = raycast(offsetOrigin1, refractedRay, surface.maxIterations / 2);

    // Upon exiting the sdf, the ray will refract once more. The IOR will also invert
    if (selfSurface.material.type == MATERIAL_REFRACTIVE) {
        vec3 transmittedRay = refract(refractedRay, selfSurface.normal, mat.ior);

        // If the angle is too small, GLSL makes the refract vector 0 to avoid sqrts of negatives
        // To avoid that, we manually compute the vector, clamping the sqrt to 0
        // Not perfect, but good enough
        if (length(transmittedRay) < EPSILON) {
            transmittedRay = mat.ior * (refractedRay + selfSurface.normal * dot(selfSurface.normal, refractedRay));
        }

        vec3 offsetOrigin2 = selfSurface.position + transmittedRay * 50.0 * EPSILON;
        Surface refractedSurface = raycast(offsetOrigin2, transmittedRay, surface.maxIterations / 2);

        refractColor = calculateColorRecursive(refractedSurface, cameraForward);

        float rayDepth = distance(offsetOrigin1, selfSurface.position);
        float survivingLight = clamp(exp(-rayDepth * mat.attenuation), 0.0, 1.0);
        refractColor = mix(mat.baseColor, refractColor, survivingLight);
    }
    else {
        refractColor = calculateColorRecursive(selfSurface, cameraForward);
    }

    color = refractColor + bpColor;
    return clamp(color, vec3(0), vec3(1));
}

vec3 calculateRefractiveRecursive(Surface surface, vec3 cameraForward) {
    vec3 color;
    Material mat = surface.material;
    vec3 light = vec3(0, -1, 0);
    float lambertianFactor = dot(light, surface.normal);
    vec3 lambertColor = clamp(lambertianFactor * mat.baseColor, vec3(0), vec3(1));

    vec3 halfVec = normalize(light + cameraForward);
    float shininess = mat.shininess;
    float specAngle = max(dot(surface.normal, halfVec), 0.0);
    float spec = pow(specAngle, shininess) * 0.8;
    vec3 bpColor = vec3(1) * spec;

    // Calculate the direction of the once refracted ray
    vec3 refractColor;
    vec3 refractedRay = refract(surface.hitRay, -surface.normal, 1.0 / mat.ior);

    // Raycast to see where the refracted ray hits itself
    vec3 offsetOrigin1 = surface.position + refractedRay * 50.0 * EPSILON;
    Surface selfSurface = raycast(offsetOrigin1, refractedRay, surface.maxIterations / 2);

    // Upon exiting the sdf, the ray will refract once more. The IOR will also invert
    if (selfSurface.material.type == MATERIAL_REFRACTIVE) {
        vec3 transmittedRay = refract(refractedRay, selfSurface.normal, mat.ior);

        // If the angle is too small, GLSL makes the refract vector 0 to avoid sqrts of negatives
        // To avoid that, we manually compute the vector, clamping the sqrt to 0
        // Not perfect, but good enough
        if (length(transmittedRay) < EPSILON) {
            transmittedRay = mat.ior * (refractedRay + selfSurface.normal * dot(selfSurface.normal, refractedRay));
        }

        vec3 offsetOrigin2 = selfSurface.position + transmittedRay * 50.0 * EPSILON;
        Surface refractedSurface = raycast(offsetOrigin2, transmittedRay, surface.maxIterations / 2);

        refractColor = calculateColorRecursive2(refractedSurface, cameraForward);

        float rayDepth = distance(offsetOrigin1, selfSurface.position);
        float survivingLight = clamp(exp(-rayDepth * mat.attenuation), 0.0, 1.0);
        refractColor = mix(mat.baseColor, refractColor, survivingLight);
    }
    else {
        refractColor = calculateColorRecursive2(selfSurface, cameraForward);
    }

    color = refractColor + bpColor;
    return clamp(color, vec3(0), vec3(1));
}

vec3 calculateColor(Surface surface, vec3 cameraForward) {
    int material = surface.material.type;
    switch (material) {
        case MATERIAL_LAMBERT:     return calculateLambert(surface);
        case MATERIAL_BLINNPHONG:  return calculateBlinnPhong(surface, cameraForward);
        case MATERIAL_GLOSSY:      return calculateGlossy(surface, cameraForward);
        case MATERIAL_REFRACTIVE:  return calculateRefractive(surface, cameraForward);
        default: return calculateBackground(surface, cameraForward);
    }
}

// To prevent recursion, "recursive" shaders get replaced with blinn-phong
vec3 calculateColorRecursive(Surface surface, vec3 cameraForward) {
    int material = surface.material.type;
    switch (material) {
        case MATERIAL_LAMBERT:     return calculateLambert(surface);
        case MATERIAL_BLINNPHONG:  return calculateBlinnPhong(surface, cameraForward);
        case MATERIAL_GLOSSY:      return calculateBlinnPhong(surface, cameraForward);
        case MATERIAL_REFRACTIVE:  return calculateRefractiveRecursive(surface, cameraForward);
        default: return calculateBackground(surface, cameraForward);
    }
}

vec3 calculateColorRecursive2(Surface surface, vec3 cameraForward) {
    int material = surface.material.type;
    switch (material) {
        case MATERIAL_LAMBERT:     return calculateLambert(surface);
        case MATERIAL_BLINNPHONG:  return calculateBlinnPhong(surface, cameraForward);
        case MATERIAL_GLOSSY:      return calculateBlinnPhong(surface, cameraForward);
        case MATERIAL_REFRACTIVE:  return calculateBlinnPhong(surface, cameraForward);
        default: return calculateBackground(surface, cameraForward);
    }
}

float calculateShadow(Surface surface, Light light, float softness) {
    vec3 surfacePoint = surface.position;
    float t = 0.05;
    float shadow = 1.0;
    float ph = 1e20;
    vec3 lightDir = light.direction;
    while (t < FAR_CLIP) {
        vec3 cur = surface.position - lightDir * t;
        float h = totalSdf(cur, NULLV, NULLV, false).distance;
        if (h < EPSILON) {
            return 0.0;
        }
        float y = h * h / (2.0 * ph);
        float d = sqrt(h * h - y * y);
        shadow = min(shadow, softness * d / max(0.0, t - y));
        ph = h;
        t += h;
    }
    return shadow;
}

vec3 calculateShadedColor(Surface surface, vec3 cameraForward) {
    vec3 outColor = vec3(0);
    vec3 shaderColor = calculateColor(surface, cameraForward);
    float lightness = 1.0;
    if (surface.material.type == MATERIAL_REFRACTIVE) {
        return shaderColor;
    }
    /*for (int i = 0; i < NUM_LIGHTS; i++) {
        Light light = getLight(i);
        lightness += calculateShadow(surface, light, 1.0);
    }*/
    lightness = clamp(lightness, 0.0, 1.0);
    return mix(shaderColor * AMBIENT_LIGHT, shaderColor, lightness);
}

void main() {
    vec3 forward = normalize(u_Ref - u_Eye);
    vec3 right = normalize(cross(forward, u_Up));
    float refDist = length(u_Ref - u_Eye);

    float verticalAngle = tan(FOV / 2.0);
    float aspectRatio = u_Dimensions.x / u_Dimensions.y;
    vec3 V = u_Up * refDist * verticalAngle;
    vec3 H = right * refDist * aspectRatio * verticalAngle;
    vec3 worldPoint = u_Ref + H * fs_Pos.x + V * fs_Pos.y;
    vec3 rayDir = normalize(worldPoint - u_Eye);

    Surface hitSurface = raycast(u_Eye, rayDir, MAX_ITERATIONS);
    vec3 color = calculateShadedColor(hitSurface, forward);

    out_Col = vec4(color, 1.0);
}
