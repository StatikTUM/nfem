TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Canvas</title>
</head>
<body>

<style>
* {
    margin: 0;
    padding: 0;
}
</style>

<div id="container"></div>
<div id="slider">
    <input type="range" min="0" max="10" value="0" step="1" class="slider" id="timestep" style="width:100%;">
    <button type="button" id="start">start</button>
    <button type="button" id="stop">stop</button>
</div>

<script src="https://unpkg.com/three@0.119.1/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.119.1/examples/js/controls/OrbitControls.js"></script>

<script src="https://unpkg.com/guify@0.12.0/lib/guify.min.js"></script>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/5.16.0/d3.min.js"></script>

<script type="text/javascript">
    let data = {{data}};

    let container = document.getElementById('container');
    let timestepSlider = document.getElementById('timestep');

    let width = window.innerWidth;
    let height = data.height;

    let nbScenes = data.frames.length;

    timestepSlider.setAttribute("max", nbScenes - 1);

    // camera
    let fov = 45;
    let aspect = width / height;
    let near = 0.1;
    let far = 1000;

    let camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
    camera.position.set(-3, 3, 3);
    camera.lookAt(new THREE.Vector3(0, 0, 0));

    camera.layers.enable(1);
    camera.layers.enable(2);
    camera.layers.enable(10);
    camera.layers.enable(20);
    camera.layers.disable(21);
    camera.layers.disable(22);
    camera.layers.enable(23);

    // renderer
    let renderer = new THREE.WebGLRenderer({ antialias: true });

    let controls = new THREE.OrbitControls(camera, renderer.domElement);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    renderer.outputEncoding = THREE.sRGBEncoding;

    container.appendChild(renderer.domElement);


    let scenes = [];
    let currentScene = data.frame || 0;

    function mod(a, b) {
        return ((a % b) + b) % b
    }

    currentScene = mod(currentScene, nbScenes);

    timestepSlider.value = currentScene;

    let materialCache = new Map();
    let tags = new Map();

    let selection = [];

    function getMaterial(color) {
        if (materialCache.has(color)) {
            return materialCache.get(color);
        }

        let material = new THREE.LineBasicMaterial({
            color: color,
        });

        materialCache.set(color, material);

        return material;
    }

    function line(scene, item) {
        let material = getMaterial(item.color || 'red');

        let pts = [];
        for (let point of item.points) {
            pts.push(new THREE.Vector3(point.y || 0, point.z || 0, point.x || 0));
        }
        let geometry = new THREE.BufferGeometry().setFromPoints(pts);

        let object = new THREE.Line(geometry, material);

        if (item.layer !== undefined) {
            object.layers.set(item.layer);
        }

        scene.add(object);
        selection.push(object);
    }

    function point(scene, item) {
        let material = new THREE.PointsMaterial({
            color: item.color || 'red',
            size: item.size || 4,
            sizeAttenuation: false,
        });

        let pts = [new THREE.Vector3(item.location.y || 0, item.location.z || 0, item.location.x || 0)];

        let geometry = new THREE.BufferGeometry().setFromPoints(pts);

        let object = new THREE.Points(geometry, material);

        if (item.layer !== undefined) {
            object.layers.set(item.layer);
        }

        scene.add(object);
        selection.push(object);
    }

    function text(scene, item) {
        if (!tags.has(item.text)) {
            // ID
            const canvas = document.createElement('canvas'); //document.getElementById("number");
            canvas.setAttribute("width", 64);
            canvas.setAttribute("height", 64);
            const ctx = canvas.getContext("2d");
            const x = 32;
            const y = 32;
            const radius = 30;
            const startAngle = 0;
            const endAngle = Math.PI * 2;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = "rgb(0, 0, 0)";
            ctx.beginPath();
            ctx.arc(x, y, radius, startAngle, endAngle);
            ctx.fill();

            ctx.strokeStyle = "rgb(255, 255, 255)";
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(x, y, radius, startAngle, endAngle);
            ctx.stroke();

            ctx.fillStyle = "rgb(255, 255, 255)";
            ctx.font = "32px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(item.text, x, y);

            // Sprite
            const numberTexture = new THREE.CanvasTexture(canvas);

            const spriteMaterial = new THREE.SpriteMaterial({
                map: numberTexture,
                alphaTest: 0.5,
                transparent: true,
                depthTest: false,
                depthWrite: false,
                sizeAttenuation: false,
            });

            tags.set(item.text, spriteMaterial);
        }

        let spriteMaterial = tags.get(item.text);

        let object = new THREE.Sprite(spriteMaterial);
        object.position.set(item.location.y || 0, item.location.z || 0, item.location.x || 0);
        object.scale.set(0.05, 0.05, 1);

        if (item.layer !== undefined) {
            object.layers.set(item.layer);
        }

        scene.add(object);
        selection.push(object);
    }

    for (let frame of data.frames) {
        let scene = new THREE.Scene();
        scene.background = new THREE.Color(0xffffff);

        scenes.push(scene);

        let grid = new THREE.GridHelper(20, 20, 0xbbbbbb, 0xdddddd);
        grid.layers.set(1);
        scene.add(grid);

        let axes = new THREE.AxesHelper(5);
        axes.layers.set(2);
        scene.add(axes);

        for (let item of frame.items) {
            switch (item.type) {
                case 'line':
                    line(scene, item);
                    break;
                case 'point':
                    point(scene, item);
                    break;
                case 'text':
                    text(scene, item);
                    break;
                case 'support':
                    {
                        const h = 0.5 * 2 / 3 * 0.5;
                        const r = 0.3 * 2 / 3 * 0.5;

                        function sub(a, b) {
                            return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
                        }

                        const p = [0, 0, 0];
                        const a = [-r, -r, -h];
                        const b = [-r, +r, -h];
                        const c = [+r, +r, -h];
                        const d = [+r, -r, -h];

                        const dx = item.location.y || 0;
                        const dy = item.location.z || 0;
                        const dz = item.location.x || 0;

                        function toVector3(point) {
                            const x = point[0] || 0;
                            const y = point[1] || 0;
                            const z = point[2] || 0;
                            return new THREE.Vector3(y, z, x);
                        }

                        const points = [p, a, b, c, d].map(toVector3);

                        for (let direction of "xyz") {
                            if (!item.direction.includes(direction)) {
                                continue;
                            }

                            const geometry = new THREE.Geometry().setFromPoints(points);
                            if (direction == "x") {
                                geometry.rotateY(-Math.PI / 2);
                            }
                            if (direction == "y") {
                                geometry.rotateZ(-Math.PI / 2);
                            }
                            geometry.translate(dx, dy, dz);
                            const wireGeometry = geometry.clone();

                            geometry.faces.push(new THREE.Face3(0, 1, 2));
                            geometry.faces.push(new THREE.Face3(0, 2, 3));
                            geometry.faces.push(new THREE.Face3(0, 3, 4));
                            geometry.faces.push(new THREE.Face3(0, 4, 1));
                            geometry.faces.push(new THREE.Face3(1, 2, 4));
                            geometry.faces.push(new THREE.Face3(3, 4, 2));

                            const material = new THREE.MeshBasicMaterial({
                                color: item.color || 'red',
                                side: THREE.DoubleSide,
                                opacity: 0.4,
                                transparent: true,
                            });

                            let object1 = new THREE.Mesh(geometry, material);
                            scene.add(object1);
                            selection.push(object1);
                            if (item.layer !== undefined) {
                                object1.layers.set(item.layer);
                            }

                            wireGeometry.faces.push(new THREE.Face3(0, 1, 2));
                            wireGeometry.faces.push(new THREE.Face3(0, 2, 3));
                            wireGeometry.faces.push(new THREE.Face3(0, 3, 4));
                            wireGeometry.faces.push(new THREE.Face3(0, 4, 1));

                            const wireMaterial = new THREE.MeshBasicMaterial({
                                color: item.color || 'red',
                                wireframe: true,
                            });

                            let object = new THREE.Mesh(wireGeometry, wireMaterial);

                            scene.add(object);
                            selection.push(object);

                            if (item.layer !== undefined) {
                                object.layers.set(item.layer);
                            }
                        }
                    }
                    break;
                case 'arrow':
                    {
                        let lx = item.location.y || 0;
                        let ly = item.location.z || 0;
                        let lz = item.location.x || 0;
                        let dx = item.direction.y || 0;
                        let dy = item.direction.z || 0;
                        let dz = item.direction.x || 0;
                        var dir = new THREE.Vector3(dx, dy, dz);

                        dir.normalize();

                        var origin = new THREE.Vector3(lx, ly, lz);
                        var length = 1;
                        var hex = 0xffff00;

                        var object = new THREE.ArrowHelper(dir, origin, length, item.color || 'red');
                        scene.add(object);
                        selection.push(object);

                        if (item.layer !== undefined) {
                            object.traverse((node) => { node.layers.set(item.layer); });
                        }
                    }
                    break;
            }
        }
    }

    function render() {
        renderer.render(scenes[currentScene], camera);
    }

    function update(index) {
        currentScene = index;
        render();
    }


    function fitCameraToSelection(camera, controls, selection, fitOffset = 1.2) {
        const box = new THREE.Box3();

        for (const object of selection) box.expandByObject(object);

        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());

        const maxSize = Math.max(size.x, size.y, size.z);
        const fitHeightDistance = maxSize / (2 * Math.atan(Math.PI * camera.fov / 360));
        const fitWidthDistance = fitHeightDistance / camera.aspect;
        const distance = fitOffset * Math.max(fitHeightDistance, fitWidthDistance);

        const direction = controls.target.clone()
            .sub(camera.position)
            .normalize()
            .multiplyScalar(distance);

        controls.maxDistance = distance * 10;
        controls.target.copy(center);

        camera.near = distance / 100;
        camera.far = distance * 100;
        camera.updateProjectionMatrix();

        camera.position.copy(controls.target).sub(direction);

        controls.update();
    }

    fitCameraToSelection(camera, controls, selection);

    d3.select('#timestep').on('input', function () {
        update(this.value);
    });

    render();



    controls.addEventListener('change', () => render());

    // gui

    let gui = new guify({
        title: null,
        theme: 'dark',
        align: 'right',
        width: 300,
        barMode: 'none',
        panelMode: 'inner',
        opacity: 0.95,
        root: container,
        open: false,
    });

    let settings = {
        grid: true,
        axes: true,
        scale: 1.0,
        undeformed: true,
        deformed: true,
        external_forces: true,
        residual_forces: false,
        nodal_results: 'None',
        element_results: 'None',
        custom: true,
        timestep_per_sec: 2.0,
        loop: true,
        reverse: false,
    };

    // gui.Register({
    //     type: 'range',
    //     label: 'scale',
    //     min: 0, max: 100, step: 1,
    //     object: settings, property: "scale",
    //     onChange: (data) => {
    //     }
    // });

    gui.Register({
        type: 'checkbox',
        label: 'grid',
        object: settings,
        property: 'grid',
        onChange: (data) => {
            if (data) {
                camera.layers.enable(1);
            } else {
                camera.layers.disable(1);
            }
            render();
        }
    });

    gui.Register({
        type: 'checkbox',
        label: 'axes',
        object: settings,
        property: 'axes',
        onChange: (data) => {
            if (data) {
                camera.layers.enable(2);
            } else {
                camera.layers.disable(2);
            }
            render();
        }
    });
  
    gui.Register({
        type: 'button',
        label: 'Zoom all',
        action: () => {
            fitCameraToSelection(camera, controls, selection);
            render();
        }
    });

    // results

    gui.Register({
        type: 'title',
        label: 'Results'
    });

    gui.Register({
        type: 'checkbox',
        label: 'undeformed',
        object: settings,
        property: 'undeformed',
        onChange: (data) => {
            if (data) {
                camera.layers.enable(10);
            } else {
                camera.layers.disable(10);
            }
            render();
        }
    });

    gui.Register({
        type: 'checkbox',
        label: 'deformed',
        object: settings,
        property: 'deformed',
        onChange: (data) => {
            if (data) {
                camera.layers.enable(20);
            } else {
                camera.layers.disable(20);
            }
            render();
        }
    });

    gui.Register({
        type: 'checkbox',
        label: 'external forces',
        object: settings,
        property: 'external_forces',
        onChange: (data) => {
            if (data) {
                camera.layers.enable(23);
            } else {
                camera.layers.disable(23);
            }
            render();
        }
    });

    gui.Register({
        type: 'select',
        label: 'nodal results',
        object: settings,
        property: 'nodal_results',
        options: ['None', 'ID'],
        onChange: (data) => {
            if (data == 'ID') {
                camera.layers.enable(21);
            } else {
                camera.layers.disable(21);
            }
            render();
        }
    });

    gui.Register({
        type: 'select',
        label: 'element results',
        object: settings,
        property: 'element_results',
        options: ['None', 'ID'],
        onChange: (data) => {
            if (data == 'ID') {
                camera.layers.enable(22);
            } else {
                camera.layers.disable(22);
            }
            render();
        }
    });

    // gui: animation

    gui.Register({
        type: 'title',
        label: 'Animation'
    });

    gui.Register({
        type: 'range',
        label: 'timesteps/sec',
        min: 0.5, max: 30, step: 0.1,
        object: settings, property: "timestep_per_sec",
        onChange: (data) => {
            clearInterval(animationTimer);
            animationTimer = setInterval(animate, 1000 / data);
        }
    });

    gui.Register({
        type: 'checkbox',
        label: 'reverse',
        object: settings,
        property: 'reverse',
        onChange: (data) => {
        }
    });

    // animation

    let animationTimer;

    function animate() {
        let b = d3.select("#timestep");
        let t = mod(+b.property("value") + (settings.reverse ? -1 : 1), +b.property("max") + 1);
        b.property("value", t);
        update(t);
    }

    d3.select("#start").on("click", function () {
        clearInterval(animationTimer);
        animationTimer = setInterval(animate, 1000 / settings.timestep_per_sec);
    });

    d3.select("#stop").on("click", function () {
        clearInterval(animationTimer);
    });


    window.addEventListener('resize', onWindowResize, false);

    function onWindowResize(){
        camera.aspect = window.innerWidth / height;
        camera.updateProjectionMatrix();

        renderer.setSize(window.innerWidth, height);

        render();
    }

    onWindowResize();
</script>

</body>
</html>
"""
