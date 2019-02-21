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
// This was originally 1000, but let's be real, we only need 30.
const float FAR_CLIP = 100.0;
const int MAX_ITERATIONS = 256;

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
const int MATERIAL_SAND = 4;
const int MATERIAL_WOOD1 = 5;
const int MATERIAL_WOOD2 = 6;
const int MATERIAL_FALLING_SAND = 7;

const float AMBIENT_LIGHT = 0.2;
const int LIGHT_DIRECTIONAL = 0;
const int LIGHT_POINT = 1;
const int NUM_LIGHTS = 3;

const int NUM_BBS = 4;

const vec3 SEED3 = vec3(0.31415, 0.6456, 0.23432);
const vec2 SEED2 = vec2(-0.42422, 0.9842);

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
};

struct Light {
    int type;
    float intensity;
    vec3 position;
    vec3 direction;
    vec3 color;
};

struct BoundingBox {
    vec3 boxCenter;
    vec3 boxRadii;
};


/* --------------------------------------------------------------------------*
 *   Utility Functions
 * --------------------------------------------------------------------------*/

float random1(vec3 p, vec3 seed) {
    return fract(sin(dot(p + seed, vec3(987.654, 123.456, 531.975))) * 85734.3545);
}

float random1(vec2 p, vec2 seed) {
    return fract(sin(dot(p + seed, vec2(127.1, 311.7))) * 43758.5453);
}

Light getLight(int index) {
    index = clamp(index, 0, NUM_LIGHTS);
    Light lights[NUM_LIGHTS];
    lights[0] = Light(LIGHT_DIRECTIONAL, 0.5, NULLV, normalize(vec3(0.7071, -0.7071, 0)), vec3(1, 0.9, 1));
    lights[1] = Light(LIGHT_DIRECTIONAL, 0.5, NULLV, normalize(vec3(-0.7071, -0.7071, 0)), vec3(1, 0.9, 1));
    lights[2] = Light(LIGHT_DIRECTIONAL, 0.3, NULLV, normalize(vec3(0, 1, 0)), vec3(0.7, 0.4, 0));
    //lights[1] = Light(LIGHT_DIRECTIONAL, 1.0, NULLV, normalize(vec3(-1, -1, 0)), vec3(1, 1, 1));
    return lights[index];
}

BoundingBox getBoundingBox(int index) {
    // 0: Full Hourglass
    // 1: Tabletop 
    // 2: Floor
    // 3: Table leg

    vec3 center = 
        index == 0 ? vec3(0, 1.5, 0) :
        index == 1 ? vec3(0, -2.37, 0) :
        index == 2 ? vec3(0, -15, 0) :
        index == 3 ? vec3(0, -8.7, 0) :
        vec3(0);
    vec3 radii =
        index == 0 ? vec3(2.82, 2.9, 2.82) :
        index == 1 ? vec3(7.01, 1.01, 7.01) :
        index == 2 ? vec3(30.01, 0.11, 30.01) :
        index == 3 ? vec3(1.01, 7.01, 1.01) :
        vec3(0);

    return BoundingBox(center, radii);
}

float dot2(vec3 v) {
    return dot(v, v);
}

float sawtooth(float p, float period) {
    return fract(p / period);
}

vec3 cosinePalette(vec3 a, vec3 b, vec3 c, vec3 d, float t) {
    return a + b * cos(6.28318 * (c * t + d));
}

float vmax(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

float smootherStep(float v1, float v2, float t) {
    t = t * t * t * (3.0 * (2.0 * t * t - 5.0 * t) + 10.0);
    return mix(v1, v2, t);
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
    return v1.x > v2.x && v1.y > v2.y && v1.z > v2.z;
}

bool vLess(vec2 v1, vec2 v2) {
    return v1.x < v2.x && v1.y < v2.y;
}

bool vGreater(vec2 v1, vec2 v2) {
    return v1.x > v2.x && v1.y > v2.y;
}

bool withinRect(vec2 point, vec2 rectCenter, vec2 rectRadii) {
    return vLess(abs(point - rectCenter), rectRadii);
}

bool withinBox(vec3 point, BoundingBox bb) {
    return vLess(abs(point - bb.boxCenter), bb.boxRadii);
}

/* --------------------------------------------------------------------------*
 *   Noise Functions
 * --------------------------------------------------------------------------*/

float brownianNoise3(vec3 pos, float frequency) {
    vec3 noisePos = pos * frequency;
    vec3 boxPos = floor(noisePos);

    // Get the noise at the corners of the cells
    float corner0 = random1(boxPos + vec3(0.0, 0.0, 0.0), SEED3);
    float corner1 = random1(boxPos + vec3(1.0, 0.0, 0.0), SEED3);
    float corner2 = random1(boxPos + vec3(0.0, 1.0, 0.0), SEED3);
    float corner3 = random1(boxPos + vec3(1.0, 1.0, 0.0), SEED3);
    float corner4 = random1(boxPos + vec3(0.0, 0.0, 1.0), SEED3);
    float corner5 = random1(boxPos + vec3(1.0, 0.0, 1.0), SEED3);
    float corner6 = random1(boxPos + vec3(0.0, 1.0, 1.0), SEED3);
    float corner7 = random1(boxPos + vec3(1.0, 1.0, 1.0), SEED3);

    // Get cubic interpolation factors
    float tx = smoothstep(0.0, 1.0, fract(noisePos.x));
    float ty = smoothstep(0.0, 1.0, fract(noisePos.y));
    float tz = smoothstep(0.0, 1.0, fract(noisePos.z));

    // Perform tricubic interpolation
    return(
        mix(
            mix(mix(corner0, corner1, tx), mix(corner2, corner3, tx), ty),
            mix(mix(corner4, corner5, tx), mix(corner6, corner7, tx), ty),
            tz
        )
    );
}

float fbm3(vec3 pos, float startingFrequency) {
    vec3 noisePos = pos * startingFrequency * 0.5;
    float total = 
        brownianNoise3(noisePos, startingFrequency * 2.0) / 2.0 +
        brownianNoise3(noisePos, startingFrequency * 4.0) / 4.0 +
        brownianNoise3(noisePos, startingFrequency * 8.0) / 8.0;

    return total / 0.875;
}

float brownianNoise2(vec2 pos, float frequency, vec2 seed) {
    vec2 noisePos = pos * frequency;
    vec2 boxPos = floor(noisePos);

    // Get the noise at the corners of the cells
    float corner0 = random1(boxPos + vec2(0.0, 0.0), SEED2 + seed);
    float corner1 = random1(boxPos + vec2(1.0, 0.0), SEED2 + seed);
    float corner2 = random1(boxPos + vec2(0.0, 1.0), SEED2 + seed);
    float corner3 = random1(boxPos + vec2(1.0, 1.0), SEED2 + seed);

    // Get cubic interpolation factors
    float tx = smoothstep(0.0, 1.0, fract(noisePos.x));
    float ty = smoothstep(0.0, 1.0, fract(noisePos.y));

    // Perform tricubic interpolation
    return(mix(mix(corner0, corner1, tx), mix(corner2, corner3, tx), ty));
}

float fbm2(vec2 pos, float startingFrequency, vec2 seed) {
    vec2 noisePos = pos * startingFrequency * 0.5;
    float total = 
        brownianNoise2(noisePos, startingFrequency * 2.0, seed) / 2.0 +
        brownianNoise2(noisePos, startingFrequency * 4.0, seed) / 4.0 +
        brownianNoise2(noisePos, startingFrequency * 8.0, seed) / 8.0;

    return total / 0.875;
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

bool intersectBoundingBox(vec3 rayOrigin, vec3 rayDir, BoundingBox bb) {
    vec3 boxCenter = bb.boxCenter;
    vec3 boxRadii = bb.boxRadii;
    if (withinBox(rayOrigin, bb)) {
        return true;
    }

    vec3 boxCorner = boxCenter - boxRadii;
    vec3 boxDimensions = boxRadii * 2.0;
    vec3 topCorner = boxCorner + boxDimensions;

    float txMin = min((boxCorner.x - rayOrigin.x) / rayDir.x, (topCorner.x  - rayOrigin.x) / rayDir.x);
    float tyMin = min((boxCorner.y - rayOrigin.y) / rayDir.y, (topCorner.y  - rayOrigin.y) / rayDir.y);
    float tzMin = min((boxCorner.z - rayOrigin.z) / rayDir.z, (topCorner.z  - rayOrigin.z) / rayDir.z);

    vec3 intX = rayOrigin + rayDir * txMin;
    vec3 intY = rayOrigin + rayDir * tyMin;
    vec3 intZ = rayOrigin + rayDir * tzMin;

    bool wrX = withinRect(intX.yz, boxCenter.yz, boxRadii.yz);
    bool wrY = withinRect(intY.xz, boxCenter.xz, boxRadii.xz);
    bool wrZ = withinRect(intZ.xy, boxCenter.xy, boxRadii.xy);

    return(
        (txMin > 0.0 && wrX) ||
        (tyMin > 0.0 && wrY) ||
        (tzMin > 0.0 && wrZ)
    );
}


/* --------------------------------------------------------------------------*
 *   SDF Primitives / Operations
 * --------------------------------------------------------------------------*/

// Sphere
float sphereSdf(vec3 p, vec3 offset, float radius) {
    return distance(p, offset) - radius;
}

// Ellipsoid
float ellipsoidSdf(vec3 p, vec3 center, vec3 radii) {
    vec3 pos = p - center;
    float k0 = length(pos / radii);
    float k1 = length(pos / (radii * radii));
    return k0 * (k0 - 1.0) / k1;
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

// Capsule
float capsuleSdf(vec3 p, vec3 center, float height, float radius) {
    vec3 pos = p - center;
    return length(vec3(pos.x, pos.y - clamp(pos.y, 0.0, height), pos.z)) - radius;
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
    float t = clamp(0.5 + 0.5 * (-d1 + d2) / smoothness, 0.0, 1.0);
    return mix(d2, d1, t) - smoothness * t * (1.0 - t);
}
float unionSdf(float d1, float d2) {
    return min(d1, d2);
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
    SDFData sdf = sdf1;
    if (sdf1.distance > sdf2) {
        sdf = SDFData(sdf2, material);
    }
    return sdf;
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

float columnSdf(vec3 p, vec3 center, vec3 origin, vec3 rayDir) {
    vec3 pCol = p;
    float pColScale = 0.224;
    pCol -= center;
    pCol = abs(pCol);
    pCol = pCol / pColScale;

    float column = unionSdf(
        unionSdf(
            unionSdf(
                capsuleSdf(pCol, vec3(0, 0.0, 0), 8.8, 0.25),
                sphereSdf(pCol, vec3(0, 6.3, 0), 0.5),
                1.2
            ),
            unionSdf(
                ellipsoidSdf(pCol, vec3(0, 1.9, 0), vec3(1, 1.6, 1)),
                torusSdf(pCol, vec3(0, 0.4, 0), 0.5, 0.2)
            ),
            1.0
        ),
        sphereSdf(pCol, vec3(0, 8.8, 0), 0.5),
        0.2
    );

    column *= pColScale;
    return column;
}

SDFData totalSdf(vec3 p, vec3 origin, vec3 rayDir, bool[NUM_BBS] boundingBoxes) {
    Material null = Material(MATERIAL_BG, NULLV, 0.0, 0.0, 0.0, 0.0);
    SDFData data = addSdf(FAR_CLIP * 5.0, null);
    float seconds = u_Time / 60.0;

    // --- Hourglass -----------------------
    float glass = FAR_CLIP * 2.0;
    float stand = FAR_CLIP * 2.0;

    if (boundingBoxes[0]) {
        float glassHull = intersectSdf(
            unionSdf(
                unionSdf(
                    coneSdf(p, vec3(0, 0.8, 0), 1.0, 1.25, 0.0),
                    sphereSdf(p, vec3(0, -0.5, 0), 1.5),
                    0.3
                ),
                unionSdf(
                    coneSdf(p, vec3(0, 2.2, 0), 1.0, 0.00, 1.25),
                    sphereSdf(p, vec3(0, 3.5, 0), 1.5),
                    0.3
                ),
                0.2
            ),
            boxSdf(p, vec3(0, 1.5, 0), vec3(1.5, 2.2, 1.5))
        );
        glass = abs(glassHull) - 0.075;
        Material glassMat = Material(MATERIAL_REFRACTIVE, vec3(0, 0.5, 0.3), 20.0, 1.0, 1.2, 1.0);
        data = addSdf(data, glass, glassMat);

        stand = unionSdf(
            unionSdf(
                coneSdf(p, vec3(0, -0.75, 0), 0.2, 2.6, 2.4),
                cylinderSdf(p, vec3(0, -1.15, 0), 0.41, 2.8),
                0.1
            ),
            unionSdf(
                coneSdf(p, vec3(0, 3.75, 0), 0.2, 2.4, 2.6),
                subtractSdf(
                    cylinderSdf(p, vec3(0, 4.15, 0), 0.4, 2.6),
                    torusSdf(p, vec3(0, 4.7, 0), 2.6, 0.6),
                    0.05
                )
            )
        );


        stand = unionSdf(
            stand,
            columnSdf(vec3(abs(p.x), p.y, abs(p.z)), vec3(1.414, 1.5, 1.414), origin, rayDir),
            0.1
        );
        Material standMat = Material(MATERIAL_GLOSSY, vec3(0.6, 0.3, 0.04), 16.0, 0.35, 1.0, 1.0);
        data = addSdf(data, stand, standMat);

        
        const float totalTime = 100.0;
        float t = clamp(seconds / totalTime, 0.0, 1.0);
        
        float bottomSandOffset = mix(-1.23, 0.0, sqrt(t));
        float bottomSand = intersectSdf(
            unionSdf(
                sphereSdf(p, vec3(0, -1.3 + bottomSandOffset, 0), 2.0),
                sphereSdf(p, vec3(0, 0.5 + bottomSandOffset, 0), 0.4),
                0.3
            ),
            glassHull + 0.13,
            0.1
        );
        Material sandMat = Material(MATERIAL_SAND, vec3(0.85, 0.6, 0.4), 0.0, 0.0, 0.0, 0.0);
        data = addSdf(data, bottomSand, sandMat);

        float topSandoffset = mix(0.0, 0.90, t * t);
        float topSand = intersectSdf(
            subtractSdf(
                sphereSdf(p, vec3(0, 3.25, 0.0), 1.8),
                sphereSdf(p, vec3(0, 4.2 - topSandoffset, 0.0), 1.7),
                0.3
            ),
            glassHull + 0.13,
            0.1
        );

        Material topSandMat = Material(MATERIAL_SAND, vec3(0.85, 0.6, 0.4), 2.0 * topSandoffset, 0.0, 0.0, 0.0);
        data = addSdf(data, topSand, topSandMat);

        float fallingSand = seconds < totalTime && seconds > 0.0 ?
            coneSdf(p, vec3(0), 1.5, 0.08, 0.02): FAR_CLIP * 5.0;
        Material fallingSandMat = Material(MATERIAL_FALLING_SAND, vec3(0.85, 0.6, 0.4), 2.0, 0.0, 0.0, 0.0);
        data = addSdf(data, fallingSand, fallingSandMat);
    }

    // --- Table ---------------------------

    if (boundingBoxes[1]) {
        Material tableMat = Material(MATERIAL_WOOD1, vec3(0.1, 0.05, 0), 12.0, 0.1, 1.0, 1.0);
        float tableTop = blendSdf(
            boxSdf(p, vec3(0, -2.37, 0), vec3(7, 1.0, 7)),
            cylinderSdf(p, vec3(0, -2.37, 0), 2.0, 7.0),
            0.5
        );
        data = addSdf(data, tableTop, tableMat);
    }

    if (boundingBoxes[3]) {
        Material legMat = Material(MATERIAL_BLINNPHONG, vec3(0.1, 0.05, 0), 12.0, 0.1, 1.0, 1.0);
        float tableLeg = cylinderSdf(p, vec3(0, -8.7, 0), 14.0, 1.0);
        data = addSdf(data, tableLeg, legMat);
    }

    // --- Floor ----------------------------

    if (boundingBoxes[2]) {
        float floor = boxSdf(p, vec3(0, -15, 0), vec3(30.0, 0.1, 30.0));
        Material floorMat = Material(MATERIAL_WOOD2, vec3(0.19, 0.07, 0.04), 4.0, 0.1, 1.0, 1.0);
        data = addSdf(data, floor, floorMat);
    }

    return data;
}


SDFData totalSdf(vec3 p, vec3 origin, vec3 rayDir) {
    bool[NUM_BBS] boundingBoxes;
    for (int i = 0; i < NUM_BBS; i++) {
        boundingBoxes[i] = withinBox(origin,getBoundingBox(i));
    }
    return totalSdf(p, origin, rayDir, boundingBoxes);
}

/*
 * Examines the entire SDF map and uses gradients to find what the
 * surface normal would be at a point in 3D space. If the point isn't
 * near a surface, the computation won't be very useful.
 */
vec3 getNormal(vec3 p) {
    vec3 dx = vec3(EPSILON, 0, 0);
    vec3 dy = vec3(0, EPSILON, 0);
    vec3 dz = vec3(0, 0, EPSILON);
    float curDist = totalSdf(p, p, vec3(0, 1, 0)).distance;

    /* Note that while I could find use the value at p + dp instead of
     * curDist, what appears to be an obscure WebGL bug/limitation (which
     * I should mention, is pure conjecture on my part, since I couldn't
     * find a single mention of this problem anywhere on the web) prevents
     * me from doing so. If a single fragment has to calculate too many
     * branches in a single pass, the entire image fails to render, with
     * no sorting of warning or error message. There are lots of conditionals
     * in totalSdf, so I want to limit my calls to that function.
     */
    float gradX = curDist - totalSdf(p - dx, p - dx, vec3(0, 1, 0)).distance;
    float gradY = curDist - totalSdf(p - dy, p - dy, vec3(0, 1, 0)).distance;
    float gradZ = curDist - totalSdf(p - dz, p - dz, vec3(0, 1, 0)).distance;

    return -normalize(vec3(gradX, gradY, gradZ));
}


/*
 * Perform the actual raycasting. This does not have to be done from
 * the camera, but except for reflections, etc., it usually is
 */
Surface raycast(vec3 origin, vec3 rayDir) {

    // We calculate all the bounding-box intersections before the raycast so we don't have
    // to do it every iteration. over 1000 bounding box computations per fragment per frame
    // really takes its toll on the browsers, and my graphics card.

    bool boundingBoxes[NUM_BBS];
    for (int i = 0; i < NUM_BBS; i++) {
        boundingBoxes[i] = intersectBoundingBox(origin, rayDir, getBoundingBox(i));
    }

    vec3 curPoint = origin + rayDir * 2.0 * EPSILON;
    Material nullMaterial;
    Surface surface = Surface(curPoint, NULLV, rayDir, nullMaterial);
    SDFData sdf;

    for(int curIteration = 0; curIteration < MAX_ITERATIONS; curIteration++) {
        sdf = totalSdf(curPoint, origin, rayDir, boundingBoxes);
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
    surface.material = Material(MATERIAL_BG, vec3(1, 0, 0), 0.0, 0.0, 0.0, 0.0);
    surface.position = curPoint; // Set for debugging purposes;
    return surface;
}


/* --------------------------------------------------------------------------*
 *   Material Calculations
 * --------------------------------------------------------------------------*/

 // Because GLSL doesn't allow for recursion, we have to redeclare a few functions
 // to fake it.
vec3 calculateColorRecursive(Surface surface, vec3 cameraForward);

vec3 calculateBackground(Surface surface, vec3 cameraForward) {
    vec3 warpOffset = vec3(cos(u_Time * 0.0015), 0.0, sin(u_Time * 0.0015));
    float fbm = fbm3(surface.hitRay + vec3(fbm3(surface.hitRay + warpOffset, 4.0)), 3.0);
    vec3 bgColor = mix(vec3(0.1, 0, 0.1), vec3(0), fbm);
    return bgColor;
}

vec3 calculateLambert(Surface surface) {
    Material mat = surface.material;
    vec3 lambertColor = vec3(0);
    vec3 color = vec3(0);
    for (int i = 0; i < NUM_LIGHTS; i++) {
        Light light = getLight(i);
        float lambFactor = clamp(dot(light.direction * light.intensity, surface.normal), AMBIENT_LIGHT, light.intensity);
        lambertColor += lambFactor * mat.baseColor * light.color;
    }

    color = lambertColor;
    return clamp(color, vec3(0), vec3(1));
}

vec3 calculateSand(Surface surface) {
    Material mat = surface.material;
    surface.material.baseColor =
        mat.baseColor + (vec3(fbm3(surface.position + vec3(0, mat.shininess, 0), 6.0) - 0.5)) * 0.15;
    return calculateLambert(surface);
}

vec3 calculateSpec(Surface surface, vec3 cameraForward) {
    Material mat = surface.material;
    float shininess = mat.shininess;
    vec3 blinnphongColor = vec3(0);
    vec3 color = vec3(0);
    vec3 totalSpec = vec3(0);

    for (int i = 0; i < NUM_LIGHTS; i++) {
        Light light = getLight(i);
        vec3 halfVec = normalize(light.direction + cameraForward);
        float specAngle = max(dot(surface.normal, halfVec), 0.0);
        float spec = pow(specAngle, shininess) * light.intensity;
        totalSpec += light.color * spec;
    }

    return totalSpec;
}

// This function is never used directly, so I'm commenting it out
// to save on compilation time
/*
vec3 calculateBlinnPhong(Surface surface, vec3 cameraForward) {
    Material mat = surface.material;
    float shininess = mat.shininess;
    vec3 lambertColor = vec3(0);
    vec3 specColor = calculateSpec(surface, cameraForward);
    vec3 color = vec3(0);

    for (int i = 0; i < NUM_LIGHTS; i++) {
        Light light = getLight(i);
        float lambFactor = clamp(dot(light.direction * light.intensity, surface.normal), AMBIENT_LIGHT, light.intensity);
        lambertColor += lambFactor * mat.baseColor * light.color;
    }

    color = lambertColor + specColor;
    return clamp(color, vec3(0), vec3(1));
}*/

// Turns out I don't need to calculate the reflection vector manually. Imagine that.
vec3 calculateGlossy(Surface surface, vec3 cameraForward) {
    Material mat = surface.material;
    float shininess = mat.shininess;
    float reflexivity = mat.reflexivity;
    vec3 lambertColor = vec3(0);
    vec3 blinnphongColor = calculateSpec(surface, cameraForward);
    vec3 glossyColor = vec3(0);

    for (int i = 0; i < NUM_LIGHTS; i++) {
        Light light = getLight(i);
        float lambFactor = clamp(dot(light.direction * light.intensity, surface.normal), AMBIENT_LIGHT, light.intensity);
        lambertColor += lambFactor * mat.baseColor * light.color;
    }

    vec3 reflectedRay = reflect(surface.hitRay, surface.normal);
    Surface reflectedSurface = raycast(surface.position, reflectedRay);
    glossyColor = reflectedSurface.material.type == MATERIAL_BG ?
        calculateBackground(reflectedSurface, cameraForward) :
        calculateLambert(reflectedSurface);

    float fresnel = clamp(1.0 - dot(surface.normal, cameraForward), 0.0, 1.0);
    float reflectionCoefficient = doubleSmoothstep(0.0, fresnel, 1.0, mat.reflexivity);

    vec3 color = 
        (AMBIENT_LIGHT * mat.baseColor + lambertColor) * (1.0 - reflectionCoefficient) + 
        glossyColor * reflectionCoefficient +
        blinnphongColor;
    return clamp(color, vec3(0), vec3(1));
}

vec3 calculateWood1(Surface surface, vec3 cameraForward) {
    Material mat = surface.material;
    vec3 p = surface.position + vec3(0, 0, floor(surface.position.z));

    float noiseP = fbm3(vec3(p.x, p.y, p.z * 20.0), 1.0);
    float noiseGradX = noiseP - fbm3(vec3(p.x + EPSILON, p.y, p.z * 20.0), 1.0);
    float noiseGradZ = noiseP - fbm3(vec3(p.x, p.y, (p.z + EPSILON) * 20.0), 1.0);
    vec3 normPerturb = normalize(vec3(noiseGradX, 0, noiseGradZ)) * 0.02;

    //mat.baseColor += noiseP * 0.05;
    surface.normal = normalize(surface.normal + normPerturb);
    surface.material = mat;
    return calculateGlossy(surface, cameraForward);
}

vec2 calculateFloorNoise(vec2 p) {
    p = p * 0.5;
    float offset = fract(p.x / 2.0) > 0.5 ? 0.0 : 3.0;
    vec2 planks = vec2(p.x, (p.y + offset) / 6.0);
    vec2 plankPos = floor(planks);
    vec2 plankFract = planks - plankPos;

    float perturbance = fbm2(p, 0.7, plankPos * 5.0);
    float woodNoise = sawtooth(p.x + perturbance, 0.225 - 0.06 * brownianNoise2(p, 0.5, plankPos));

    float test = smootherStep(0.0, 1.0, clamp(((plankFract.x - 0.01) * 40.0), 0.0, 1.0));
    test *= smootherStep(1.0, 0.0, clamp((plankFract.x - 0.965) * 40.0, 0.0, 1.0));
    test *= smootherStep(0.0, 1.0, clamp(plankFract.y * 240.0, 0.0, 1.0));
    test *= smootherStep(1.0, 0.0, clamp((plankFract.y - 0.991665) * 240.0, 0.0, 1.0));
    float rawNoise = (woodNoise * 0.3 + 0.5) * test;
    float rawColor = woodNoise * 0.5 + 0.5 - fbm2(vec2((p.x + perturbance) * 15.0, p.y), 2.0, SEED2 + plankPos) * 0.25;
    rawColor *= test;
    return vec2(rawColor * rawColor, rawNoise * rawNoise);
}

vec3 calculateWood2(Surface surface, vec3 cameraForward) {
    vec2 p = surface.position.xz;
    vec2 woodP = calculateFloorNoise(p);

    float noiseP = woodP.y;
    float noiseX = calculateFloorNoise(p + vec2(EPSILON, 0)).y;
    float noiseZ = calculateFloorNoise(p + vec2(0, EPSILON)).y;
    const float DEPTH = 0.5;

    vec3 p1 = vec3(p.x, noiseP * DEPTH, p.y);
    vec3 p2 = vec3(p.x + EPSILON, noiseX * DEPTH, p.y);
    vec3 p3 = vec3(p.x, noiseZ * DEPTH, p.y + EPSILON);

    surface.material.baseColor = cosinePalette(
        vec3(0.228, 0.428, 0.148),
        vec3(0.293, 0.321, -0.032),
        vec3(0.478, 0.108, 1.000),
        vec3(0.618, 1.568, 0.667),
        woodP.x
    ) * 0.7;
    surface.normal = normalize(cross(p2 - p1, p3 - p1));
    
    Material mat = surface.material;
    float shininess = mat.shininess;
    vec3 lambertColor = vec3(0);
    vec3 specColor = calculateSpec(surface, cameraForward);
    vec3 color = vec3(0);

    for (int i = 0; i < NUM_LIGHTS; i++) {
        Light light = getLight(i);
        float lambFactor = clamp(dot(light.direction * light.intensity, surface.normal), AMBIENT_LIGHT, light.intensity);
        lambertColor += lambFactor * mat.baseColor * light.color;
    }

    color = lambertColor + specColor * 0.15;
    return clamp(color, vec3(0), vec3(1));
}

// I literally spent days working on this, and then I learned that GLSL has a built in
// function that automatically calculates refraction vectors, so that's how my week's been
// going.
vec3 calculateRefractive(Surface surface, vec3 cameraForward) {
    vec3 color;
    Material mat = surface.material;
    vec3 light = vec3(0, -1, 0);

    vec3 bpColor = calculateSpec(surface, cameraForward);

    // Calculate the direction of the once refracted ray
    vec3 refractColor;
    vec3 refractedRay = refract(surface.hitRay, -surface.normal, 1.0 / mat.ior);

    // Raycast to see where the refracted ray hits itself
    vec3 offsetOrigin1 = surface.position + refractedRay * 50.0 * EPSILON;
    Surface selfSurface = raycast(offsetOrigin1, refractedRay);
    int hitMaterialType = selfSurface.material.type;

    // Upon exiting the sdf, the ray will refract once more. The IOR will also invert
    vec3 transmittedRay = refract(refractedRay, selfSurface.normal, mat.ior);

    // If the angle is too small, GLSL makes the refract vector 0 to avoid sqrts of negatives
    // To avoid that, we manually compute the vector, clamping the sqrt to 0
    // Not perfect, but good enough
    transmittedRay = length(transmittedRay) > EPSILON ?
        transmittedRay :
        mat.ior * (refractedRay + selfSurface.normal * dot(selfSurface.normal, refractedRay));

    vec3 offsetOrigin2 = selfSurface.position + transmittedRay * 50.0 * EPSILON;
    Surface refractedSurface = raycast(offsetOrigin2, transmittedRay);

    refractColor = calculateColorRecursive(refractedSurface, cameraForward);

    float rayDepth = distance(offsetOrigin1, selfSurface.position);
    float survivingLight = clamp(exp(-rayDepth * mat.attenuation), 0.0, 1.0);
    vec3 mixColor = hitMaterialType == MATERIAL_REFRACTIVE ? refractColor : selfSurface.material.baseColor;
    refractColor = mix(mat.baseColor, mixColor, survivingLight);

    color = refractColor + bpColor;
    return clamp(color, vec3(0), vec3(1));
}

vec3 calculateRefractiveRecursive(Surface surface, vec3 cameraForward) {
    vec3 color;
    Material mat = surface.material;
    vec3 bpColor = calculateSpec(surface, cameraForward);

    // Calculate the direction of the once refracted ray
    vec3 refractColor;
    vec3 refractedRay = refract(surface.hitRay, -surface.normal, 1.0 / mat.ior);

    // Raycast to see where the refracted ray hits itself
    vec3 offsetOrigin1 = surface.position + refractedRay * 50.0 * EPSILON;
    Surface selfSurface = raycast(offsetOrigin1, refractedRay);
    int hitMaterialType = selfSurface.material.type;

    // Upon exiting the sdf, the ray will refract once more. The IOR will also invert
    vec3 transmittedRay = refract(refractedRay, selfSurface.normal, mat.ior);

    // If the angle is too small, GLSL makes the refract vector 0 to avoid sqrts of negatives
    // To avoid that, we manually compute the vector, clamping the sqrt to 0
    // Not perfect, but good enough
    transmittedRay = length(transmittedRay) > EPSILON ?
        transmittedRay :
        mat.ior * (refractedRay + selfSurface.normal * dot(selfSurface.normal, refractedRay));

    vec3 offsetOrigin2 = selfSurface.position + transmittedRay * 50.0 * EPSILON;
    Surface refractedSurface = raycast(offsetOrigin2, transmittedRay);

    refractColor = refractedSurface.material.type == MATERIAL_BG ?
        calculateBackground(refractedSurface, cameraForward) :
        calculateLambert(refractedSurface);

    float rayDepth = distance(offsetOrigin1, selfSurface.position);
    float survivingLight = clamp(exp(-rayDepth * mat.attenuation), 0.0, 1.0);
    vec3 mixColor = hitMaterialType == MATERIAL_REFRACTIVE ? refractColor : selfSurface.material.baseColor;
    refractColor = mix(mat.baseColor, mixColor, survivingLight);

    color = refractColor + bpColor;
    return clamp(color, vec3(0), vec3(1));
}

vec3 calculateFallingSand(Surface surface, vec3 cameraForward) {
    vec3 pos = surface.position;
    float grain = brownianNoise3(pos + vec3(0, u_Time * 0.01, 0), 100.0);
    surface.material.baseColor += vec3(grain * 0.5);
    return calculateLambert(surface);
}

vec3 calculateColor(Surface surface, vec3 cameraForward) {
    switch (surface.material.type) {
        case MATERIAL_LAMBERT:     return calculateLambert(surface);
        //case MATERIAL_BLINNPHONG:  return calculateBlinnPhong(surface, cameraForward);
        case MATERIAL_GLOSSY:      return calculateGlossy(surface, cameraForward);
        case MATERIAL_REFRACTIVE:  return calculateRefractive(surface, cameraForward);
        case MATERIAL_SAND:        return calculateSand(surface);
        case MATERIAL_FALLING_SAND:return calculateFallingSand(surface, cameraForward);
        case MATERIAL_WOOD1:       return calculateWood1(surface, cameraForward);
        case MATERIAL_WOOD2:       return calculateWood2(surface, cameraForward);
        default: return calculateBackground(surface, cameraForward);
    }
}

// To prevent recursion, "recursive" shaders get replaced with blinn-phong
vec3 calculateColorRecursive(Surface surface, vec3 cameraForward) {
    switch (surface.material.type) {
        case MATERIAL_REFRACTIVE:  return calculateRefractiveRecursive(surface, cameraForward);
        case MATERIAL_SAND:        return calculateSand(surface);
        case MATERIAL_FALLING_SAND:return calculateFallingSand(surface, cameraForward);
        case MATERIAL_BG:          return calculateBackground(surface, cameraForward);
        default: return calculateLambert(surface);
    }
}

float calculateShadow(Surface surface, Light light, float softness) {
    vec3 surfacePoint = surface.position;
    float t = 0.05;
    float shadow = 1.0;
    float ph = 1e20;
    vec3 lightDir = light.direction;

    bool boundingBoxes[NUM_BBS];
    for (int i = 0; i < NUM_BBS; i++) {
        BoundingBox bb = getBoundingBox(i);
        bb.boxRadii += vec3(1);
        boundingBoxes[i] = intersectBoundingBox(surface.position, -lightDir, bb);
    }

    int i = 0;
    while(t < FAR_CLIP && i < MAX_ITERATIONS) {
        vec3 cur = surface.position - lightDir * t;
        float h = totalSdf(cur, cur, vec3(0, 1, 0), boundingBoxes).distance;
        if (h < EPSILON) {
            return 0.0;
        }
        float y = h * h / (2.0 * ph);
        float d = sqrt(h * h - y * y);
        shadow = min(shadow, softness * d / max(0.0, t - y));
        ph = h;
        t += h;
        i++;
    }
    return shadow * light.intensity;
}

vec3 calculateShadedColor(Surface surface, vec3 cameraForward) {
    vec3 outColor = vec3(0);
    vec3 shaderColor = calculateColor(surface, cameraForward);
    float lightness = 0.0;
    for (int i = 0; i < NUM_LIGHTS; i++) {
        Light light = getLight(i);
        lightness += calculateShadow(surface, light, 15.0);
    }
    lightness = clamp(lightness, 0.0, 1.0);
    int matType = surface.material.type;
    vec3 shadowedColor = (matType == MATERIAL_REFRACTIVE || matType == MATERIAL_BG) ?
        shaderColor :
        mix(shaderColor * AMBIENT_LIGHT * vec3(1, 1, 0.3), shaderColor, lightness);
    float distance = length(surface.position);
    return mix(shadowedColor, calculateBackground(surface, cameraForward), clamp((distance - 28.0) / 5.0, 0.0, 1.0)); 
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

    Surface hitSurface = raycast(u_Eye, rayDir);
    vec3 shadedColor = calculateShadedColor(hitSurface, forward);

    out_Col = vec4(shadedColor, 1.0);
}
