/*Modified from VantaJS, by Alex Kotlar copyright 2020. Original license follows:*/
/*Copyright 2020 Teng Bao

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/* from helpers.js*/
function ri(start, end) {
  if (start == null) start = 0;
  if (end == null) end = 1;
  return Math.floor(start + (Math.random() * ((end - start) + 1)));
}

function getBrightness(threeColor) {
  return (0.299 * threeColor.r) + (0.587 * threeColor.g) + (0.114 * threeColor.b);
}

/*pruned/extended _base.js + vanta.net.js, focused on performance improvement, drops cpu usage by 75%, reduces memory usage, and introduces hover effects*/
class Viz {
  constructor(userOptions = {}) {
    if (!THREE.WebGLRenderer) {
      console.error("ThreeJS has not been loaded, or WebGL is unsupported");
      return;
    }

    this.options = Object.assign({
      scale: 1,
      scaleMobile: 1,
      color: 0xff3f81,
      backgroundColor: 0xfffffff,
      points: 10,
      maxDistance: 20,
      spacing: 15,
      showDots: true
    }, userOptions);

    this.el = document.querySelector(this.options.el);
    if (!this.el) {
      console.error(`Cannot find ${this.options.el}`);
      return;
    }

    if (!this.el.style.position == 'absolute') {
      this.el.style.position = 'absolute';
    }
    this.el.style.opacity = 0;
    this.hidden = true;

    this.mouse = { "x": 0, "y": 0, "rawY": 0, "updated": false, "updatedCount": -1, "ran": false };

    this.highlightColor = new THREE.Color('purple');
    this.cachedColor = new THREE.Color(0x000000);
    this.options.color = new THREE.Color(this.options.color);
    this.options.backgroundColor = new THREE.Color(this.options.backgroundColor);
    this.diffColor = this.options.color.clone().sub(this.options.backgroundColor);
    this.colorB = getBrightness(new THREE.Color(this.options.color));
    this.bgB = getBrightness(new THREE.Color(this.options.backgroundColor));

    this.elOffset = this.el.offsetTop;
    this.elOnscreen = false;
    this.isScrolling = false;
    this.resizeTimeout = null;
    this.postInit = false;
    this.points = [];

    this.animationInterval = null;
    this.animationLoop = this.animationLoop.bind(this);

    window.requestAnimationFrame(() => {
      this.renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: true,
      });

      this.el.appendChild(this.renderer.domElement);
    });

    const intersectionThreshold = 0.6;
    const intersectionCallback = (entries) => {
      if (entries.length > 1) {
        console.warn("should be observing a single element, ignoring all but first");
      }

      // entries[0].isIntersecting incorrect in firefox
      this.elOnscreen = entries[0].intersectionRatio > intersectionThreshold;
      this.interval = 1000 / 16;
      if (this.elOnscreen) {
        if(!this.postInit)  {
          try {
            window.requestAnimationFrame(() => {
              this.init();
              this.listen();
              this.then = Date.now();
              this.animationLoop(24);
              this.postInit = true;
            });
          } catch (e) {
            if (this.renderer && this.renderer.domElement) {
              this.el.removeChild(this.renderer.domElement)
            }
            log.error(e);
            return
          }

          return;
        }

        this.animationLoop(24);
        return;
      }

      clearInterval(this.animationInterval);
    };

    let observer = new IntersectionObserver(intersectionCallback, { threshold: intersectionThreshold });

    window.requestAnimationFrame(() => observer.observe(this.renderer.domElement));

  }

  listen() {
    this.elOffset = this.el.offsetTop;

    this.isScrolling = false;
    this.resizeTimeout = null;

    window.addEventListener('resize', (e) => this.resize(e));
    window.addEventListener('scroll', () => {
      if (this.isScrolling) {
        window.clearTimeout(this.isScrolling);
      }

      this.isScrolling = setTimeout(() => this.isScrolling = null, 100);
    });

    let timeout;
    window.addEventListener('mousemove', (e) => {
      if (timeout) {
        clearTimeout(timeout);
      }

      timeout = setTimeout(() => {
        this.onMouseMove2(e)
        timeout = null;
      }, this.mouse.dontshow ? 32 : 4);
    }, false);

    this.mouse.dontshow = false;

    // TODO: generalize this
    const d = document.getElementById('hero-content');
    const n = document.getElementById('hail-navbar');

    d.onmouseover = () => {
      if (timeout) {
        clearTimeout(timeout);
      }
      this.mouse.updated = false;
      this.mouse.updatedCount = 0;
      this.mouse.dontshow = true;
    }

    d.onmouseout = () => {
      if (timeout) {
        clearTimeout(timeout);
      }
      this.mouse.updated = true;
      this.mouse.updatedCount = 0;
      this.mouse.dontshow = false;
    }

    n.onmouseover = () => {
      if (timeout) {
        clearTimeout(timeout);
      }
      this.mouse.updated = false;
      this.mouse.updatedCount = 0;
      this.mouse.dontshow = true;
    }

    n.onmouseout = () => {
      if (timeout) {
        clearTimeout(timeout);
      }
      this.mouse.updated = true;
      this.mouse.updatedCount = 0;
      this.mouse.dontshow = false;
    }
  }

  resize(e) {
    if (this.resizeTimeout) {
      clearTimeout(this.resizeTimeout);
    }
    this.resizeTimeout = setTimeout(() => {
      if (this.camera) {
        this.camera.aspect = this.el.offsetWidth / this.el.offsetHeight;
        if (typeof this.camera.updateProjectionMatrix === "function") {
          this.camera.updateProjectionMatrix()
        }
      }
      if (this.renderer) {
        this.renderer.setSize(this.el.offsetWidth, this.el.offsetHeight)
        this.renderer.setPixelRatio(window.devicePixelRatio)
      }

      this.resizeTimeout = null;
    }, 100);
  }

  animationLoop(tInterval =  24) {
    this.animationInterval =  window.setInterval(() => {
      if(this.startedAnimation) {
        return;
      }

      const now = Date.now();
      const delta = now - this.then;

      if (this.elOnscreen && !this.isScrolling) {
        if (delta > this.interval) {
          this.onUpdate()
          if (this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera)
            // this.renderer.setClearColor(this.options.backgroundColor, this.options.backgroundAlpha)
          }
        }
      }

      if(this.hidden) {
        this.startedAnimation = true;
        const started =  Date.now();

        window.requestAnimationFrame(() => {
          this.el.style.opacity = "1";
          this.hidden = false;
          this.startedAnimation = false;
          console.info("done", Date.now() - started);
        });

        this.then = now - 1000 - (delta % this.interval);
      } else {
        this.then = now - (delta % this.interval);
      }
    }, tInterval);
  }


  onMouseMove2(e) {
    if (!this.elOnscreen || this.mouse.dontshow) {
      return;
    }

    if (!this.mouse.ran) {
      this.mouse.ran = true;
      return;
    }

    if (!this.rayCaster) {
      this.rayCaster = new THREE.Raycaster()
    }

    const ox = e.pageX;
    const oy = e.pageY - this.elOffset;
    const x = (ox / this.el.offsetWidth) * 2 - 1;

    const y = - (oy / this.el.offsetHeight) * 2 + 1;

    if (x !== this.mouse.x || y !== this.mouse.y) {
      this.mouse.x = x;
      this.mouse.y = y;
      this.mouse.updated = true;
      this.mouse.updatedCount = 0;

      this.rayCaster.setFromCamera(new THREE.Vector2(this.mouse.x, this.mouse.y), this.camera);
    }
  }

  genPoint(x, y, z) {
    const geometry = new THREE.SphereGeometry(0.25, 12, 12);
    const material = new THREE.MeshLambertMaterial({
      color: this.options.color,
      // blending: THREE.AdditiveBlending,
      transparent: true,
      opacity: .2
    });
    const sphere = new THREE.Mesh(geometry, material);
    sphere.position.set(x, y, z);
    sphere.r = 0.00025 * ((Math.random() * 4) - 2); // rotation rate, larger is faster
    return sphere;
  }

  init() {
    const group = new THREE.Group();
    group.position.set(0, 0, 0);

    let { points, spacing } = this.options;

    const numPoints = points * points * 2;
    this.linePositions = new Float32Array(numPoints * numPoints * 3);
    this.lineColors = new Float32Array(numPoints * numPoints * 3);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(this.linePositions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(this.lineColors, 3));
    geometry.computeBoundingSphere();
    geometry.setDrawRange(0, 0);
    const material = new THREE.LineBasicMaterial({
      vertexColors: THREE.VertexColors,
      // blending: THREE.AdditiveBlending,
      transparent: true,
      alphaTest: .1,
      opacity: .2
    });

    this.linesMesh = new THREE.LineSegments(geometry, material)
    this.linesMesh.renderOrder = 2;
    group.add(this.linesMesh);

    for (let i = 0; i <= points; i++) {
      for (let j = 0; j <= points; j++) {
        const y = ri(-3, 3)
        const x = ((i - (points / 2)) * spacing) + ri(-5, 5);
        let z = ((j - (points / 2)) * spacing) + ri(-5, 5);
        if (i % 2) { z += spacing * 0.5 };

        const p1 = this.genPoint(x, y - ri(5, 15), z);
        const p2 = this.genPoint(x + ri(-5, 5), y + ri(5, 15), z + ri(-5, 5));
        group.add(p1, p2);
        this.points.push(p1, p2);
      }
    }

    this.renderer.setSize(this.el.offsetWidth, this.el.offsetHeight)
    this.renderer.setPixelRatio(window.devicePixelRatio)

    const ambience = new THREE.AmbientLight(0xffffff, 0.75);
    this.camera = new THREE.PerspectiveCamera(25, this.el.offsetWidth / (this.el.offsetHeight), .01, 10000);

    this.camera.position.set(50, 100, 150);
    this.camera.lookAt(0, 0, 0);

    this.scene = new THREE.Scene();
    this.scene.add(this.camera)
    this.scene.add(ambience);
    this.scene.add(group);
  }

  onUpdate() {
    let vertexpos = 0;
    let colorpos = 0;
    let numConnected = 0;

    let dist, distToMouse, lineColor, p, p2, ang;
    let affected1 = 0;
    for (let i = 0; i < this.points.length; i++) {
      p = this.points[i];

      if (this.rayCaster) {
        if (this.mouse.updated) {
          distToMouse = (12 - this.rayCaster.ray.distanceToPoint(p.position)) * 0.25;
          if (distToMouse > 1) {
            affected1 = 1;
            p.material.color = this.highlightColor;
          } else {
            affected1 = 0;
            p.material.color = this.options.color;
          }
        }
        else if (p.material.color !== this.options.color) {
          p.material.color = this.options.color;
        }
      }

      if (p.r !== 0) {
        ang = Math.atan2(p.position.z, p.position.x) + p.r;
        dist = Math.sqrt((p.position.z ** 2) + (p.position.x ** 2));
        p.position.x = dist * Math.cos(ang);
        p.position.z = dist * Math.sin(ang);
      }

      for (let j = i; j < this.points.length; j++) {
        p2 = this.points[j]
        dist = Math.sqrt(((p.position.x - p2.position.x) ** 2) + ((p.position.y - p2.position.y) ** 2) + ((p.position.z - p2.position.z) ** 2))
        if (dist < this.options.maxDistance) {
          if (affected1) {
            lineColor = this.highlightColor;
          } else {
            let alpha = ((1.0 - (dist / this.options.maxDistance)));
            if (alpha < 0) {
              alpha = 0;
            } else if (alpha > 1) {
              alpha = 1;
            }

            lineColor = this.options.backgroundColor.clone().lerp(this.options.color, alpha);
          }
          this.linePositions[vertexpos++] = p.position.x;
          this.linePositions[vertexpos++] = p.position.y;
          this.linePositions[vertexpos++] = p.position.z;
          this.linePositions[vertexpos++] = p2.position.x;
          this.linePositions[vertexpos++] = p2.position.y;
          this.linePositions[vertexpos++] = p2.position.z;


          this.lineColors[colorpos++] = lineColor.r;
          this.lineColors[colorpos++] = lineColor.g;
          this.lineColors[colorpos++] = lineColor.b;
          this.lineColors[colorpos++] = lineColor.r;
          this.lineColors[colorpos++] = lineColor.g;
          this.lineColors[colorpos++] = lineColor.b;

          numConnected++;
        }
      }
    }

    this.linesMesh.geometry.setDrawRange(0, numConnected * 2);
    this.linesMesh.geometry.attributes.position.needsUpdate = true;
    this.linesMesh.geometry.attributes.color.needsUpdate = true;
  }
}

export default Viz;
