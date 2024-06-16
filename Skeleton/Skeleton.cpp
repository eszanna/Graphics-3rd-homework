//=============================================================================================

#include "framework.h"

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;

	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation
	void main() {
		//texture2D(textureUnit, texcoord).a = 0.5f;
		fragmentColor = texture2D(textureUnit, texcoord)*vec4(1.0f,1.0f,1.0f,1.0f); 
	}
)";
vec3 operator/(const vec3& a, const vec3& b) {
	return vec3(a.x / b.x, a.y / b.y, a.z / b.z);
}

struct Material {
	vec3 ka, kd, ks; //ka = ambient light reflection, kd = diffuse reflection, ks = specular reflection
	float  shininess;
	bool reflective = false; 
	bool refractive = false;
	vec3 kappa;
	vec3 n;

	Material(vec3 _kd, vec3 _ks, float _shininess) : kd(_kd), ks(_ks), shininess(_shininess)
	{ ka =  3.0f * kd; } //rucskos anyagok ambiens visszaverodesi tenyezoje a diffuz tenyezojenek a haromszorosa

	vec3 Fresnel() {
		vec3 one(1, 1, 1);
		vec3 F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
		return F0;
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	bool out = true;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
	Ray(vec3 _start, vec3 _dir, bool _out) { start = _start; dir = normalize(_dir); out = _out; }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Cylinder : public Intersectable {
		vec3 center;
		vec3 direction;
		float radius;
		float height;

		Cylinder(const vec3& _center, const vec3& _direction, float _radius, float _height, Material* _material) {
			center = _center;
			direction = normalize(_direction);
			radius = _radius;
			height = _height;
			material = _material;
		}

		Hit intersect(const Ray& ray) {
			Hit hit;
			vec3 oc = ray.start - center;
			float a = dot(ray.dir, ray.dir) - pow(dot(ray.dir, direction), 2);
			float b = 2.0f * (dot(ray.dir, oc) - dot(ray.dir, direction) * dot(oc, direction));
			float c = dot(oc, oc) - pow(dot(oc, direction), 2) - radius * radius;
			float discr = b * b - 4.0f * a * c;
			if (discr < 0) 
				return hit;

			float sqrt_discr = sqrtf(discr);
			float t1 = (-b + sqrt_discr) / 2.0f / a;
			float t2 = (-b - sqrt_discr) / 2.0f / a;
			if (t1 <= 0) 
				return hit;

			hit.t = (t2 > 0) ? t2 : t1;
			vec3 hitPos = ray.start + ray.dir * hit.t;
			float y = dot(hitPos - center, direction);
			if (y < 0 || y > height) 
				return Hit();

			hit.position = hitPos;
			hit.normal = normalize(hit.position - center - direction * y);
			hit.material = material;

			return hit;
		}
};

struct Cone : public Intersectable {
	vec3 apex;
	vec3 direction;
	float angle;
	float height;

	Cone(const vec3& _apex, const vec3& _direction, float _angle, float _height, Material* _material) {
		apex = _apex;
		direction = normalize(_direction);
		angle = _angle;
		material = _material;
		height = _height;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 oc = ray.start - apex;
		float k = tan(angle);
		float a = dot(ray.dir, ray.dir) - (1 + k * k) * pow(dot(ray.dir, direction), 2);
		float b = 2.0f * (dot(ray.dir, oc) - (1 + k * k) * dot(ray.dir, direction) * dot(oc, direction));
		float c = dot(oc, oc) - (1 + k * k) * pow(dot(oc, direction), 2);
		float discr = b * b - 4.0f * a * c;
		if (discr < 0)
			return hit;

		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0)
			return hit;

		hit.t = (t2 > 0) ? t2 : t1;
		vec3 hitPos = ray.start + ray.dir * hit.t;
		float y = dot(hitPos - apex, direction);
		if (y < 0 || y > height)
			return Hit();

		hit.position = hitPos;
		hit.normal = normalize((hit.position - apex) - direction * y);
		hit.material = material;

		return hit;
	}
};


struct Plane : public Intersectable {
	vec3 point;
	vec3 normal;

	Plane(const vec3& _point, const vec3& _normal) {
		point = _point;
		normal = normalize(_normal);
	}
	public:
		Hit intersect(const Ray& ray) {
			Hit hit;
			float denom = dot(normal, ray.dir);
			vec3 kd1(0, 0.1, 0.3), kd2(0.3, 0.3, 0.3), ks(0, 0, 0);
			Material* material1 = new Material(kd1, ks, 50);
			Material* material2 = new Material(kd2, ks, 50);

			if (fabs(denom) > 0.0001f) { // avoid division by zero
				float t = dot(point - ray.start, normal) / denom;
				if (t >= 0) {
					vec3 hitPos = ray.start + ray.dir * t;
					// Check if the hit point is within the 20x20 meter square
					if (hitPos.x >= -10 && hitPos.x <= 10 && hitPos.z >= -10 && hitPos.z <= 10) {
						hit.t = t;
						hit.position = hitPos;
						hit.normal = normal;
						// calculate the color based on the hit position
						int x = floor(hit.position.x);
						int z = floor(hit.position.z);
						if ((x + z) % 2 == 0) {
							hit.material = material2; // white
						}
						else {
							hit.material = material1; // blue
						}
					}
				}
			}
			return hit;
		}
	};

class Camera {
	vec3 eye, lookat, right, up;
public:

	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}

	void moveCamera(float angle) {
		// Calculate the new eye position based on the angle and direction
		float newX =  cos(M_PI / 4) * eye.x + sin(M_PI / 4) * eye.z;
		float newZ = -sin(M_PI / 4) * eye.x + cos(M_PI / 4) * eye.z;

		eye.x = newX;
		eye.z = newZ;

		float fov = M_PI / 4;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vec3(0.0f,1.0f,0.0f), w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
		
		set(eye, lookat, up, fov);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

const float epsilon = 0.0001f;

class Scene {
public:
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(0, 1, 4), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(1.0f, 1.0f, 1.0f), Le(2.0f, 2.0f, 2.0f);

		lights.push_back(new Light(lightDirection, Le));

		objects.push_back(new Plane(vec3(0, -1, 0), vec3(0, 1, 0)));

		//Yellow diffuse-specular
		vec3 kd1(0.3f, 0.2f, 0.1f), ks1(2.0f, 2.0f, 2.0f);
		Material* yellow = new Material(kd1, ks1, 50);
		yellow->reflective = false;
		yellow->refractive = false;

		//Magenta diffuse-specular
		vec3 kd2(0.3f, 0.0f, 0.2f), ks2(2.0f, 2.0f, 2.0f);
		Material* magenta = new Material(kd2, ks2, 20);
		magenta->reflective = false;
		magenta->refractive = false;

		//Cyan diffuse-specular
		vec3 kd3(0.1f, 0.2f, 0.3f), ks3(2.0f, 2.0f, 2.0f);
		Material* cyan = new Material(kd3, ks3, 100);
		cyan->reflective = false;
		cyan->refractive = false;

		//Gold reflective
		vec3 kd4(0.17f, 0.35f, 0.5f), ks4(3.1f, 2.7f, 1.9f);
		Material* gold = new Material(kd4, ks4, 0);
		gold->reflective = true;
		gold->refractive = false;
		gold->n = vec3(0.17f, 0.35f, 1.5f); //toresmutato
		gold->kappa= vec3(3.1f, 2.7f, 1.9f); //kioltasi tenyezo

		//Transparent
		vec3 kd5(0.0f, 0.0f, 0.0f), ks5(1.0f, 1.0f, 1.0f);
		Material* transparent = new Material(kd5, ks5, 0);
		transparent->reflective = true;
		transparent->refractive = true;
		transparent->n = vec3(1.3f, 1.3f, 1.3f);
		transparent->kappa = vec3(0.0f, 0.0f, 0.0f);
		
			//Cyan
		objects.push_back(new Cone(vec3(0.0f, 1.0f, 0.0f), vec3(-0.1f, -1.0f, -0.05f), 0.2f , 2.0f, cyan));
			//Magenta
		objects.push_back(new Cone(vec3(0.0f, 1.0f, 0.8f), vec3(0.2f, -1.0f, 0.0f), 0.2f, 2.0f, magenta));
			//Yellow
		objects.push_back(new Cylinder(vec3(-1.0f, -1.0f, 0.0f), vec3(0.0f, 1.0f, 0.1f), 0.3f, 2.0f, yellow));
			//Gold
		objects.push_back(new Cylinder(vec3(1.0f, -1.0f, 0.0f), vec3(0.1f, 1.0f, 0.0f), 0.3f, 2.0f, gold));
			 //Transparent
		objects.push_back(new Cylinder(vec3(0.0f, -1.0f, -0.8f), vec3(-0.2f, 1.0f, -0.1f), 0.3f, 2.0f, transparent));	
	}

	//calculates every pixel with trace
	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}
	
	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 5) 
			return La;

		Hit hit = firstIntersect(ray);
		if (hit.t < 0) 
			return La;

		vec3 outRadiance(0,0,0);

		//arany henger
		if (hit.material->reflective == true) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1, 1, 1);
			
			vec3 F = hit.material->Fresnel() + (one - hit.material->Fresnel()) * pow(1 - cosa, 5);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth+1) * F;
		}

		//atlatszo henger
		if (hit.material->refractive == true) {
			float ior = 1.3f; //viz toresmutatoja

			vec3 n(0.0f, 0.0f, 0.0f);

			if (ray.out) {
				n = vec3(ior, ior, ior); //toresmutato
				ior = 1.3f;
			}
			else {
				//toresmutato 1/ior ha kifele jon a sugar
				n = vec3(1.0f / 1.3f , 1.0f / 1.3f, 1.0f / 1.3f);
				ior = 1 / 1.3f;
			}

			vec3 refractionDir = refract(ray.dir, hit.normal, ior);

			if (length(refractionDir) > 0) {

				Ray refractRay(hit.position - hit.normal * epsilon, refractionDir, !ray.out);

				float cosa = -dot(ray.dir, hit.normal);
				vec3 one(1, 1, 1);
				vec3 F = hit.material->Fresnel() + (one - hit.material->Fresnel()) * pow(1 - cosa, 5);
				outRadiance = outRadiance + trace(refractRay, depth+1) * (vec3(1, 1, 1) - F);
			}
		}

		if(!hit.material->reflective && !hit.material->refractive) { //diffuse-specular

			outRadiance = hit.material->ka * La; //direct light hit

			for (Light* light : lights) {
				Ray shadow(hit.position + hit.normal * epsilon, light->direction);
				float cosTheta = dot(hit.normal, light->direction);

				if (cosTheta > 0 && !shadowIntersect(shadow)) {
					outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;

					vec3 halfWay = normalize(-ray.dir + light->direction);
					float cosDelta = dot(hit.normal, halfWay);
					if(cosDelta > 0)
						outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);	
				}
			}
		}
		return outRadiance;
	}

	vec3 refract(vec3 V, vec3 N, float ns) {
		float cosa = -dot(V, N);
		float discriminant = 1 - (1 - cosa * cosa) / ns / ns;
		if (discriminant < 0)
			return vec3(0.0f, 0.0f, 0.0f); // Total internal reflection
		
		return V / ns + N * (cosa / ns - sqrt(discriminant));
	}

	//reflective materials
	vec3 reflect(vec3 I, vec3 N) {
		return I - N * dot(N, I) * 2.0;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};
Scene scene;

FullScreenTexturedQuad* fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'a') {
		printf("Rotate\n");
		scene.camera.moveCamera((M_PI/ 4));
		std::vector<vec4> image(windowWidth * windowHeight);
		scene.render(image); // render the scene again
		delete fullScreenTexturedQuad;
		fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image); // update the texture
		glutPostRedisplay();
	}
}

void onKeyboardUp(unsigned char key, int pX, int pY) {

}

void onMouse(int button, int state, int pX, int pY) {
}
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	glutPostRedisplay();
}
