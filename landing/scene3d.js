(function () {
  const host = document.querySelector('[data-crimetor-scene3d]');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  if (!host || reducedMotion.matches) return;
  const clamp = (value) => Math.max(0, Math.min(1, value));
  const initialProgress = Number(host.dataset.progress);
  let state = {
    progress: Number.isFinite(initialProgress) ? clamp(initialProgress) : 0,
    scenario: host.dataset.scenario === 'fight' ? 'fight' : 'vandalism',
    character: host.dataset.character === 'jacket' ? 'jacket' : 'hoodie'
  };
  let scene, renderer, resizeObserver, renderScene, resizeScene, canvas;
  let stopped = false;

  function fallback() {
    if (stopped) return;
    stopped = true;
    host.dataset.ready = 'false';
    resizeObserver?.disconnect();
    window.removeEventListener('resize', renderAtSize);
    reducedMotion.removeEventListener('change', onMotionPreference);
    canvas?.removeEventListener('webglcontextlost', onContextLost);
    canvas?.remove();

    const resources = new Set();
    scene?.traverse((object) => {
      if (object.geometry) resources.add(object.geometry);
      if (object.material) resources.add(object.material);
      if (object.shadow) resources.add(object.shadow);
    });
    resources.forEach((resource) => resource.dispose());
    renderer?.dispose();
    renderScene = resizeScene = resizeObserver = renderer = canvas = scene = null;
  }

  function onContextLost(event) {
    event.preventDefault();
    fallback();
  }

  function onMotionPreference(event) {
    if (event.matches) fallback();
  }

  function draw(resize = false) {
    if (stopped || !renderScene) return;
    try {
      if (resize) resizeScene();
      renderScene();
    } catch {
      fallback();
    }
  }

  function renderAtSize() {
    draw(true);
  }

  function setState(next) {
    if (!next || typeof next !== 'object') return;
    const value = Number(next.progress);
    const updated = {
      progress: Number.isFinite(value) ? clamp(value) : 0,
      scenario: next.scenario === 'fight' ? 'fight' : 'vandalism',
      character: next.character === 'jacket' ? 'jacket' : 'hoodie'
    };
    if (updated.progress === state.progress && updated.scenario === state.scenario && updated.character === state.character) return;
    state = updated;
    draw();
  }

  reducedMotion.addEventListener('change', onMotionPreference);
  window.CrimetorScene3D = { setState };

  import('https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js').then((T) => {
    if (stopped) return;
    scene = new T.Scene();
    scene.background = new T.Color(0x37434a);
    const camera = new T.PerspectiveCamera(38, 1, 0.1, 100);
    renderer = new T.WebGLRenderer({ antialias: true });
    canvas = renderer.domElement;
    canvas.setAttribute('aria-hidden', 'true');
    canvas.addEventListener('webglcontextlost', onContextLost);
    Object.assign(canvas.style, { display: 'block', width: '100%', height: '100%' });
    host.appendChild(canvas);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = T.PCFSoftShadowMap;
    renderer.toneMapping = T.ACESFilmicToneMapping;
    scene.add(new T.HemisphereLight(0xc9e8ff, 0x766158, 1.55));
    const key = new T.DirectionalLight(0xffe7ce, 2.2);
    key.position.set(-3, 7, 4);
    key.target.position.set(0, 1.8, -3.2);
    key.castShadow = true;
    key.shadow.mapSize.set(1536, 1536);
    Object.assign(key.shadow.camera, { left: -7, right: 7, top: 6, bottom: -6, far: 22 });
    key.shadow.camera.updateProjectionMatrix();
    key.shadow.normalBias = 0.025;
    scene.add(key, key.target);
    const material = (color, roughness = 0.8) => new T.MeshStandardMaterial({ color, roughness });
    const wall = material(0xaab3ad, 0.92), floorMat = material(0x8b8275, 0.88);
    const boardMat = material(0x263f3d), deskMat = material(0x8b6247), metal = material(0x475763, 0.45);
    const room = new T.Group();
    scene.add(room);

    function roundedBox(size, radius) {
      radius = Math.max(0, Math.min(radius, ...size.map((value) => value * 0.49)));
      const geometry = new T.BoxGeometry(1, 1, 1, 8, 8, 8);
      const position = geometry.attributes.position;
      const normal = geometry.attributes.normal;
      const inner = new T.Vector3(...size).multiplyScalar(0.5).addScalar(-radius);
      const vertex = new T.Vector3();
      const inset = new T.Vector3();
      const bevel = new T.Vector3();
      for (let i = 0; i < position.count; i++) {
        vertex.fromBufferAttribute(position, i);
        for (const axis of ['x', 'y', 'z']) {
          const direction = Math.sign(vertex[axis]);
          const distance = Math.abs(vertex[axis]) * 4;
          inset[axis] = direction * Math.min(1, distance) * inner[axis];
          bevel[axis] = direction * Math.max(0, distance - 1);
        }
        bevel.normalize();
        vertex.copy(inset).addScaledVector(bevel, radius);
        position.setXYZ(i, vertex.x, vertex.y, vertex.z);
        normal.setXYZ(i, bevel.x, bevel.y, bevel.z);
      }
      return geometry;
    }

    function box(parent, size, pos, mat, radius = 0) {
      const geo = radius ? roundedBox(size, radius) : new T.BoxGeometry(...size);
      const mesh = new T.Mesh(geo, mat);
      mesh.position.set(...pos);
      mesh.castShadow = mesh.receiveShadow = true;
      parent.add(mesh);
      return mesh;
    }
    function plane(parent, size, pos, rot, mat) {
      const mesh = new T.Mesh(new T.PlaneGeometry(...size), mat);
      mesh.position.set(...pos);
      mesh.rotation.set(...rot);
      mesh.receiveShadow = true;
      parent.add(mesh);
      return mesh;
    }
    plane(room, [10, 12], [0, 0, 1], [-Math.PI / 2, 0, 0], floorMat);
    box(room, [10, 5.2, 0.16], [0, 2.55, -5], wall);
    plane(room, [10, 5.2], [-5, 2.55, 0], [0, Math.PI / 2, 0], wall);
    plane(room, [10, 5.2], [5, 2.55, 0], [0, -Math.PI / 2, 0], wall);
    box(room, [9.8, 0.16, 0.08], [0, 0.08, -4.87], metal);
    box(room, [4.78, 1.96, 0.11], [0, 2.65, -4.9], metal, 0.035);
    box(room, [4.62, 1.8, 0.09], [0, 2.65, -4.82], boardMat, 0.015);
    box(room, [4.78, 0.095, 0.25], [0, 1.70, -4.78], metal, 0.03);
    const paneMat = material(0x7cafb3, 0.38);
    for (const x of [-3.55, 3.55]) {
      box(room, [1.55, 1.8, 0.1], [x, 2.7, -4.87], paneMat);
      box(room, [1.68, 0.1, 0.14], [x, 3.65, -4.82], metal);
      box(room, [1.68, 0.09, 0.15], [x, 1.77, -4.82], metal);
      box(room, [1.68, 0.06, 0.12], [x, 2.63, -4.82], metal);
      box(room, [0.06, 1.92, 0.12], [x, 2.7, -4.82], metal);
      box(room, [0.1, 1.92, 0.14], [x - 0.79, 2.7, -4.82], metal);
      box(room, [0.1, 1.92, 0.14], [x + 0.79, 2.7, -4.82], metal);
    }
    for (const x of [-2.5, 0, 2.5]) {
      const desk = new T.Group();
      desk.position.set(x, 0, 0.1);
      box(desk, [1.9, 0.16, 0.9], [0, 1.05, 0], deskMat, 0.05);
      for (const dx of [-0.76, 0.76]) for (const z of [-0.31, 0.31])
        box(desk, [0.08, 1.02, 0.08], [dx, 0.52, z], metal);
      room.add(desk);
    }

    const vector = (x, y, z) => new T.Vector3(x, y, z);
    const up = vector(0, 1, 0);
    const forward = vector(0, 0, 1);
    const direction = new T.Vector3();
    function sphere(parent, radius, pos, mat, scale) {
      const mesh = new T.Mesh(new T.SphereGeometry(radius, 16, 12), mat);
      mesh.position.set(...pos);
      if (scale) mesh.scale.set(...scale);
      mesh.castShadow = true;
      parent.add(mesh);
      return mesh;
    }
    function cylinder(parent, radii, length, pos, mat) {
      const mesh = new T.Mesh(new T.CylinderGeometry(...radii, length, 24), mat);
      mesh.position.set(...pos);
      mesh.castShadow = true;
      parent.add(mesh);
      return mesh;
    }

    function limb(parent, radius, mat) {
      return cylinder(parent, [radius * 0.89, radius], 1, [0, 0, 0], mat);
    }

    function pose(mesh, start, end) {
      direction.subVectors(end, start);
      mesh.position.copy(start).lerp(end, 0.5);
      mesh.scale.y = direction.length();
      mesh.quaternion.setFromUnitVectors(up, direction.normalize());
    }

    const graffiti = new T.Group();
    graffiti.position.set(0, 2.62, -4.758);
    room.add(graffiti);
    const strokeSegments = 84;
    const strokeSides = 8;
    function stroke(points, color, radius) {
      const curve = new T.CatmullRomCurve3(points.map(([x, y]) => vector(x, y, 0)));
      const geometry = new T.TubeGeometry(curve, strokeSegments, radius, strokeSides, false);
      geometry.scale(1, 1, 0.11);
      const pigment = new T.MeshBasicMaterial({ color, toneMapped: false });
      const mesh = new T.Mesh(geometry, pigment);
      const start = sphere(graffiti, radius, curve.getPointAt(0).toArray(), pigment, [1, 1, 0.11]);
      const tip = sphere(graffiti, radius, [0, 0, 0], pigment, [1, 1, 0.11]);
      start.castShadow = tip.castShadow = false;
      graffiti.add(mesh);
      return { curve, mesh, start, tip };
    }

    function reveal(stroke, amount) {
      const segments = Math.floor(strokeSegments * amount);
      stroke.mesh.geometry.setDrawRange(0, segments * strokeSides * 6);
      stroke.start.visible = stroke.tip.visible = segments > 0;
      stroke.curve.getPointAt(segments / strokeSegments, stroke.tip.position);
    }

    const paint = stroke([[1.32, -0.24], [0.93, 0.40], [0.50, -0.13], [0.12, 0.39],
      [-0.33, -0.13], [-0.73, 0.32], [-1.16, -0.30]], 0xff594b, 0.103);
    const accent = stroke([[-0.98, -0.56], [-0.47, -0.42], [0.05, -0.55],
      [0.61, -0.38], [1.07, -0.43]], 0xff826b, 0.068);
    const accentStart = accent.curve.getPointAt(0);
    const paintEnd = paint.curve.getPointAt(1);
    const bracket = new T.MeshBasicMaterial({ color: 0xff5b53, transparent: true, depthWrite: false, toneMapped: false });
    const detection = new T.Group();
    detection.position.z = 0.025;
    for (const x of [-1, 1]) for (const y of [-1, 1]) {
      plane(detection, [0.25, 0.028], [x * 1.425, y * 0.74, 0], [0, 0, 0], bracket);
      plane(detection, [0.028, 0.23], [x * 1.55, y * 0.625, 0], [0, 0, 0], bracket);
    }
    graffiti.add(detection);

    function createStudent(style, colors = {}) {
      const actor = new T.Group();
      const isHoodie = style === 'hoodie';
      const clothes = material(colors.clothes ?? (isHoodie ? 0xb5523c : 0x3267a8));
      const trim = material(colors.trim ?? (isHoodie ? 0x77352d : 0xf0c55d));
      const skin = material(colors.skin ?? 0xe6aa81);
      const cord = material(colors.cord ?? 0xe7c5a4);
      const pants = material(colors.pants ?? 0x354350);
      const shoes = material(colors.shoes ?? 0x28323a);
      const hair = material(colors.hair ?? 0x382723);
      const eyes = material(0x26242a), whites = material(0xf5e7dc);
      const torso = new T.Mesh(
        isHoodie ? new T.CapsuleGeometry(0.46, 0.58, 6, 12) : roundedBox([1.02, 1.12, 0.70], 0.17),
        clothes
      );
      torso.position.y = isHoodie ? 1.59 : 1.53;
      torso.scale.z = isHoodie ? 0.76 : 1;
      torso.castShadow = torso.receiveShadow = true;
      actor.add(torso);

      if (isHoodie) {
        box(actor, [0.54, 0.26, 0.08], [0, 1.37, 0.305], trim, 0.085);
        box(actor, [0.42, 0.12, 0.055], [0, 1.37, 0.353], clothes, 0.04);
        for (const x of [-0.115, 0.115]) box(actor, [0.028, 0.30, 0.035], [x, 1.98, 0.286], cord, 0.012);
      } else {
        box(actor, [0.68, 0.20, 0.22], [0, 2.09, -0.24], trim, 0.065);
        box(actor, [0.53, 0.30, 0.09], [0, 1.48, -0.367], trim, 0.035);
        box(actor, [0.075, 0.62, 0.045], [0, 1.51, -0.37], clothes, 0.018);
        box(actor, [0.86, 0.105, 0.045], [0, 1.13, -0.365], trim, 0.025);
        const stripeA = box(actor, [0.47, 0.085, 0.045], [-0.18, 1.86, -0.379], trim, 0.022);
        const stripeB = box(actor, [0.47, 0.085, 0.045], [0.18, 1.86, -0.379], trim, 0.022);
        stripeA.rotation.z = 0.26;
        stripeB.rotation.z = -0.26;
        box(actor, [0.20, 0.22, 0.045], [0.31, 1.51, -0.382], cord, 0.025);
        box(actor, [0.27, 0.17, 0.06], [0.34, 1.98, -0.34], trim, 0.035);
      }

      const head = new T.Group();
      head.position.y = 2.51;
      actor.add(head);
      if (isHoodie) {
        sphere(head, 0.35, [0, 0, -0.145], clothes, [1.05, 1.08, 0.74]);
      } else {
        sphere(head, 0.30, [0.015, -0.015, -0.09], hair, [1.13, 0.78, 0.84]);
        sphere(head, 0.22, [-0.13, 0.16, -0.075], hair, [1.22, 0.47, 0.78]);
      }
      sphere(head, 0.305, [0, -0.025, 0.11], skin, [0.87, 1.04, 0.80]);
      sphere(head, 0.246, [0, 0.18, 0.13], hair, [0.96, 0.38, 0.78]);
      sphere(head, 0.071, [0, -0.068, 0.33], skin, [0.79, 0.8, 1]);
      box(head, [0.10, 0.025, 0.026], [0, -0.16, 0.305], trim, 0.009);
      for (const x of [-1, 1]) {
        sphere(head, 0.068, [x * 0.258, -0.071, 0.145], skin, [0.75, 1.12, 0.68]);
        sphere(head, 0.057, [x * 0.112, 0.026, 0.323], whites, [0.98, 1, 0.46]);
        sphere(head, 0.031, [x * 0.108, 0.026, 0.345], eyes, [0.94, 1, 0.44]);
        const brow = box(head, [0.092, 0.028, 0.03], [x * 0.113, 0.108, 0.321], hair, 0.012);
        brow.rotation.z = x * -0.16;
      }

      const legs = [-1, 1].map((side) => ({
        side,
        leg: limb(actor, 0.17, pants),
        shoe: box(actor, [0.37, 0.18, 0.53], [0, 0, 0], shoes, 0.07),
        sole: box(actor, [0.37, 0.06, 0.54], [0, 0, 0], cord, 0.025)
      }));
      const arms = [-1, 1].map((side) => {
        const sleeve = isHoodie ? clothes : trim;
        const upper = limb(actor, 0.17, sleeve), forearm = limb(actor, 0.145, sleeve);
        const elbow = sphere(actor, 0.16, [0, 0, 0], sleeve);
        const hand = sphere(actor, 0.13, [0, 0, 0], skin);
        return { side, upper, forearm, elbow, hand, shoulder: vector(side * 0.39, 2.07, 0) };
      });
      return { actor, head, legs, arms, cord, shoes };
    }

    function createCan(actor, actorYaw, cord, shoes) {
      const can = new T.Group();
      actor.add(can);
      const sprayDistance = vector(0.24, 0, 0.28);
      can.rotation.y = Math.atan2(-sprayDistance.x, -sprayDistance.z) - actorYaw;
      cylinder(can, [0.102, 0.102], 0.36, [0, 0.015, 0], material(0xed6253, 0.43));
      cylinder(can, [0.106, 0.106], 0.087, [0, -0.044, 0], cord);
      cylinder(can, [0.106, 0.106], 0.028, [0, -0.165, 0], metal);
      cylinder(can, [0.073, 0.10], 0.044, [0, 0.196, 0], metal);
      box(can, [0.087, 0.052, 0.123], [0, 0.24, 0.033], shoes, 0.021);
      sphere(can, 0.023, [0, 0.24, 0.096], cord, [1, 1, 0.5]);
      return { can, sprayDistance, nozzleOffset: vector(0, 0.24, 0.10).applyQuaternion(can.quaternion) };
    }

    const vandalism = new T.Group();
    room.add(vandalism);
    const vandalists = {
      hoodie: createStudent('hoodie'),
      jacket: createStudent('jacket')
    };
    for (const rig of Object.values(vandalists)) {
      rig.actor.position.z = -4.02;
      rig.actor.rotation.y = -2.2;
      vandalism.add(rig.actor);
      rig.can = createCan(rig.actor, rig.actor.rotation.y, rig.cord, rig.shoes);
    }

    const fightScene = new T.Group();
    room.add(fightScene);
    const fighters = [
      createStudent('hoodie', { clothes: 0xe58a43, trim: 0x963f32, pants: 0x344b70, hair: 0x35241f }),
      createStudent('jacket', { clothes: 0x397db1, trim: 0xd6e5e7, pants: 0x4a4854, hair: 0x2b2525 })
    ];
    const fighterMarkers = [];
    for (let index = 0; index < fighters.length; index++) {
      const rig = fighters[index];
      rig.side = index === 0 ? 1 : -1;
      rig.actor.position.set(index === 0 ? -1.45 : 1.45, 0, 0.8);
      rig.actor.rotation.y = index === 0 ? 0.48 : -0.48;
      fightScene.add(rig.actor);
      const marker = new T.Group();
      marker.visible = false;
      for (const x of [-1, 1]) for (const y of [-1, 1]) {
        plane(marker, [0.22, 0.03], [x * 0.55, 1.45 + y * 1.36, 0], [0, 0, 0], bracket);
        plane(marker, [0.03, 0.25], [x * 0.67, 1.45 + y * (1.36 - 0.125), 0], [0, 0, 0], bracket);
      }
      const markerDepth = 0.84, markerYaw = rig.actor.rotation.y;
      marker.position.set(-Math.tan(markerYaw) * markerDepth, 0, markerDepth);
      marker.rotation.y = -markerYaw;
      rig.actor.add(marker);
      fighterMarkers.push(marker);
    }

    const mistMaterial = new T.MeshBasicMaterial({ color: 0xffbea7, toneMapped: false });
    const mist = Array.from({ length: 12 }, (_, index) => {
      const particle = sphere(room, 0.012 + (index % 3) * 0.006, [0, 0, 0], mistMaterial);
      particle.castShadow = false;
      return particle;
    });

    const cctv = new T.Group();
    cctv.position.set(-3.73, 4.31, -4.53);
    const lensDirection = vector(1.32, 2.14, -3.97).sub(cctv.position).normalize();
    cctv.quaternion.setFromUnitVectors(forward, lensDirection);
    room.add(cctv);
    const housing = material(0xe5e5db, 0.44), bezel = material(0x212e36, 0.38);
    box(cctv, [0.74, 0.55, 1.02], [0, 0, 0], housing, 0.115);
    box(cctv, [0.76, 0.14, 1.06], [0, 0.236, 0.078], housing, 0.063);
    box(cctv, [0.637, 0.438, 0.13], [0, -0.022, 0.47], bezel, 0.054);
    for (const z of [-0.23, -0.13, -0.03, 0.07]) {
      box(cctv, [0.019, 0.112, 0.034], [-0.361, -0.073, z], metal, 0.008);
    }
    function lensRing(radii, length, z, mat) {
      cylinder(cctv, radii, length, [0, -0.015, z], mat).rotation.x = Math.PI / 2;
    }
    lensRing([0.207, 0.217], 0.095, 0.541, metal);
    lensRing([0.197, 0.201], 0.065, 0.597, bezel);
    sphere(cctv, 0.149, [0, -0.015, 0.619], material(0x244d62, 0.18), [1, 1, 0.25]);
    sphere(cctv, 0.090, [0.011, -0.021, 0.646], material(0x19323e, 0.17), [1, 1, 0.13]);
    const glare = new T.MeshBasicMaterial({ color: 0xcde9ed, toneMapped: false });
    sphere(cctv, 0.032, [-0.046, 0.043, 0.652], glare, [1.25, 0.8, 0.07]);
    const led = new T.MeshBasicMaterial({ color: 0xa2df92, toneMapped: false });
    sphere(cctv, 0.022, [0.242, -0.117, 0.542], led, [1, 1, 0.4]);
    const mount = vector(-4.16, 4.46, -4.85);
    const joint = cctv.localToWorld(vector(0, -0.12, -0.41));
    box(room, [0.35, 0.46, 0.16], mount.toArray(), metal, 0.05);
    pose(limb(room, 0.089, bezel), mount, joint);

    resizeScene = function () {
      const w = Math.max(1, host.clientWidth), h = Math.max(1, host.clientHeight);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
      renderer.setSize(w, h, false);
    };
    const ease = (start, end) => {
      const amount = clamp((state.progress - start) / (end - start));
      return amount * amount * (3 - 2 * amount);
    };
    const closePosition = cctv.position.clone().addScaledVector(lensDirection, 1.91).add(vector(0, 0.18, 0.80));
    const closeTarget = cctv.position.clone().addScaledVector(lensDirection, 0.19);
    const widePosition = vector(-2.32, 3.36, 5.42);
    const wideTarget = vector(0.13, 2.14, -3.27);
    const fightWidePosition = vector(-2.32, 3.38, 6.25);
    const fightWideTarget = vector(0.04, 1.55, -0.18);
    const look = new T.Vector3(), mark = new T.Vector3(), sprayOrigin = new T.Vector3();
    const gaze = new T.Vector3(), elbow = new T.Vector3(), wrist = new T.Vector3();
    const hip = new T.Vector3(), ankle = new T.Vector3();
    const restingWrist = vector(0.39, 1.56, 0.33);

    renderScene = function () {
      const { progress, scenario, character } = state;
      const wide = ease(0, 0.43);
      const targetPosition = scenario === 'fight' ? fightWidePosition : widePosition;
      const targetLook = scenario === 'fight' ? fightWideTarget : wideTarget;
      camera.position.copy(closePosition).lerp(targetPosition, wide);
      camera.lookAt(look.copy(closeTarget).lerp(targetLook, wide));

      const first = ease(0.37, 0.55), second = ease(0.571, 0.68);
      const vandalismActive = scenario === 'vandalism';
      const paintProgress = vandalismActive ? first : 0;
      const accentProgress = vandalismActive ? second : 0;
      reveal(paint, paintProgress);
      reveal(accent, accentProgress);
      if (progress <= 0.55) paint.curve.getPointAt(first, mark);
      else if (progress >= 0.571) accent.curve.getPointAt(second, mark);
      else mark.copy(paintEnd).lerp(accentStart, ease(0.55, 0.571));
      mark.add(graffiti.position);

      const reach = ease(0.28, 0.37) * (1 - ease(0.68, 0.76));
      const stepping = progress <= 0.55 ? first : second;
      const stride = 0.105 * Math.sin(7 * Math.PI * stepping) * reach;
      vandalism.visible = vandalismActive;
      for (const [style, rig] of Object.entries(vandalists)) {
        rig.actor.visible = vandalismActive && character === style;
        if (!rig.actor.visible) continue;
        rig.actor.position.x = mark.x + 0.92 + 0.23 * ease(0.68, 0.77);
        for (const { side, leg, shoe, sole } of rig.legs) {
          const lift = Math.max(0, side * stride) * 0.42;
          ankle.set(side * 0.23, 0.15 + lift, side * stride);
          hip.set(side * 0.185, 1.06, 0);
          pose(leg, hip, ankle);
          shoe.position.set(ankle.x, 0.113 + lift, ankle.z + 0.089);
          sole.position.set(ankle.x, 0.035 + lift, ankle.z + 0.089);
        }
        const leftArm = rig.arms[0], rightArm = rig.arms[1];
        elbow.set(-0.49, 1.68, 0.07);
        wrist.set(-0.36, 1.43, 0.23);
        pose(leftArm.upper, leftArm.shoulder, elbow);
        pose(leftArm.forearm, elbow, wrist);
        leftArm.elbow.position.copy(elbow);
        leftArm.hand.position.set(wrist.x, wrist.y, wrist.z + 0.01);

        sprayOrigin.copy(mark).add(rig.can.sprayDistance);
        rig.actor.worldToLocal(wrist.copy(sprayOrigin)).sub(rig.can.nozzleOffset);
        wrist.lerp(restingWrist, 1 - reach);
        elbow.set(0.38 + reach * 0.06, 1.68 + reach * (0.28 + (mark.y - 2.62) * 0.43), 0.12 + reach * 0.27);
        pose(rightArm.upper, rightArm.shoulder, elbow);
        pose(rightArm.forearm, elbow, wrist);
        rightArm.elbow.position.copy(elbow);
        rightArm.hand.position.copy(wrist);
        rightArm.hand.position.x += 0.092;
        rig.can.can.position.copy(wrist);
        rig.actor.worldToLocal(gaze.copy(mark).lerp(cctv.position, ease(0.71, 0.81)));
        gaze.sub(rig.head.position);
        rig.head.rotation.y = Math.max(-0.54, Math.min(0.55, Math.atan2(gaze.x, gaze.z)));
        rig.head.rotation.x = Math.max(-0.25, Math.min(0.25, -gaze.y * 0.31));
      }

      fightScene.visible = !vandalismActive;
      const approach = ease(0.30, 0.48);
      const impact = ease(0.45, 0.56);
      const shove = impact * (1 - 0.7 * ease(0.58, 0.70));
      const settle = ease(0.66, 0.72);
      const recoil = ease(0.55, 0.69) * (1 - 0.18 * settle);
      const brace = ease(0.45, 0.55) * (1 - 0.35 * settle);
      for (const rig of fighters) {
        if (vandalismActive) continue;
        const side = rig.side;
        rig.actor.position.x = side > 0
          ? -1.45 + 0.74 * approach - 0.09 * recoil
          : 1.45 - 0.74 * approach + 0.20 * recoil;
        rig.actor.position.z = side > 0 ? 0.8 : 0.8 - 0.11 * recoil;
        rig.actor.rotation.z = side > 0 ? -0.065 * impact * (1 - 0.45 * settle) : -0.10 * recoil;
        for (const { side: legSide, leg, shoe, sole } of rig.legs) {
          const retreatStep = side < 0 && legSide > 0 ? recoil * 0.14 : 0;
          ankle.set(legSide * 0.27 + retreatStep, 0.15, legSide * 0.08 - (side < 0 ? recoil * 0.16 : 0));
          hip.set(legSide * 0.20, 1.06, 0);
          pose(leg, hip, ankle);
          shoe.position.set(ankle.x, 0.113, ankle.z + 0.09);
          sole.position.set(ankle.x, 0.035, ankle.z + 0.09);
        }
        for (const arm of rig.arms) {
          if (side > 0 && arm.side > 0) {
            elbow.set(0.43, 1.91, 0.32);
            wrist.set(0.42, 1.75, 0.29 + shove * 0.56);
          } else if (side < 0) {
            elbow.set(arm.side * 0.37, 1.89, 0.29 + brace * 0.08);
            wrist.set(arm.side * 0.15, 1.79, 0.43 + brace * 0.08);
          } else {
            elbow.set(-0.40, 1.83, 0.15);
            wrist.set(-0.36, 1.58, 0.29);
          }
          pose(arm.upper, arm.shoulder, elbow);
          pose(arm.forearm, elbow, wrist);
          arm.elbow.position.copy(elbow);
          arm.hand.position.set(wrist.x, wrist.y, wrist.z + 0.035);
        }
        rig.head.rotation.y = side < 0 ? -0.38 * brace : 0;
      }

      const spraying = vandalismActive && ((first > 0 && first < 1) || (second > 0 && second < 1));
      mist.forEach((particle, index) => {
        particle.visible = spraying;
        if (!spraying) return;
        const distance = ((progress * 87 + index * 0.163) % 1) * 0.82 + 0.09;
        particle.position.copy(mark).add(vandalists[character].can.sprayDistance).lerp(mark, distance);
        particle.position.x += Math.sin(index * 2.4) * distance * 0.062;
        particle.position.y += Math.cos(index * 2.4) * distance * 0.069;
      });
      bracket.opacity = ease(0.59, 0.72);
      detection.visible = vandalismActive && bracket.opacity > 0;
      for (const marker of fighterMarkers) marker.visible = !vandalismActive && bracket.opacity > 0;
      led.color.setHex(progress >= 0.67 ? 0xff655b : 0xa2df92);
      renderer.render(scene, camera);
    };
    renderAtSize();
    if (stopped) return;
    host.dataset.ready = 'true';
    if (typeof ResizeObserver === 'function') {
      resizeObserver = new ResizeObserver(renderAtSize);
      resizeObserver.observe(host);
    } else {
      window.addEventListener('resize', renderAtSize, { passive: true });
    }
  }).catch(fallback);
})();
