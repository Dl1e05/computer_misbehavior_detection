(function () {
  const host = document.querySelector('[data-crimetor-scene3d]');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  if (!host || reducedMotion.matches) return;
  const clamp = (value) => Math.max(0, Math.min(1, value));
  let progress = clamp(Number(window.CrimetorSceneProgress) || 0);
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

  reducedMotion.addEventListener('change', onMotionPreference);
  window.CrimetorScene3D = { setProgress(value) {
    const next = clamp(Number(value) || 0);
    if (next === progress) return;
    progress = next;
    draw();
  } };

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

    const actor = new T.Group();
    actor.position.z = -4.02;
    actor.rotation.y = -2.2;
    room.add(actor);
    const skin = material(0xe6aa81), hoodie = material(0xb5523c), hoodTrim = material(0x77352d);
    const cord = material(0xe7c5a4), pants = material(0x354350), shoes = material(0x28323a);
    const hair = material(0x382723), eyes = material(0x26242a), whites = material(0xf5e7dc);
    const torso = new T.Mesh(new T.CapsuleGeometry(0.39, 0.62, 8, 16), hoodie);
    torso.position.y = 1.59;
    torso.scale.z = 0.79;
    torso.castShadow = torso.receiveShadow = true;
    actor.add(torso);
    box(actor, [0.41, 0.22, 0.09], [0, 1.37, 0.266], hoodTrim, 0.03);
    for (const x of [-0.10, 0.10]) box(actor, [0.026, 0.28, 0.035], [x, 1.98, 0.277], cord, 0.009);
    sphere(actor, 0.23, [0, 2.16, 0.02], hoodTrim, [1.25, 0.5, 1]);

    const head = new T.Group();
    head.position.y = 2.51;
    actor.add(head);
    sphere(head, 0.365, [0, 0, -0.075], hoodie, [0.99, 1.08, 0.93]);
    sphere(head, 0.32, [0, 0, 0.042], hoodTrim, [0.98, 1.08, 0.85]);
    sphere(head, 0.298, [0, -0.025, 0.11], skin, [0.86, 1.04, 0.79]);
    sphere(head, 0.251, [0, 0.18, 0.13], hair, [0.93, 0.38, 0.81]);
    sphere(head, 0.067, [0, -0.056, 0.33], skin, [0.79, 0.83, 1]);
    box(head, [0.089, 0.022, 0.025], [0.01, -0.151, 0.305], hoodTrim, 0.008);
    for (const x of [-1, 1]) {
      sphere(head, 0.063, [x * 0.252, -0.066, 0.14], skin, [0.75, 1.15, 0.66]);
      sphere(head, 0.051, [x * 0.106, 0.028, 0.317], whites, [0.97, 1, 0.45]);
      sphere(head, 0.028, [x * 0.103, 0.028, 0.338], eyes, [0.94, 1, 0.43]);
      const brow = box(head, [0.08, 0.025, 0.029], [x * 0.108, 0.101, 0.314], hair, 0.01);
      brow.rotation.z = x * -0.13;
    }

    const legs = [-1, 1].map((side) => ({
      side,
      leg: limb(actor, 0.164, pants),
      shoe: box(actor, [0.35, 0.17, 0.52], [0, 0, 0], shoes, 0.065),
      sole: box(actor, [0.35, 0.059, 0.53], [0, 0, 0], cord, 0.024)
    }));
    pose(limb(actor, 0.145, hoodie), vector(-0.33, 2.10, 0), vector(-0.49, 1.68, 0.07));
    pose(limb(actor, 0.125, hoodie), vector(-0.49, 1.68, 0.07), vector(-0.36, 1.43, 0.23));
    sphere(actor, 0.112, [-0.36, 1.44, 0.24], skin);
    const shoulder = vector(0.33, 2.10, 0);
    const upper = limb(actor, 0.147, hoodie), forearm = limb(actor, 0.126, hoodie);
    const elbow = new T.Vector3(), wrist = new T.Vector3();
    const restingWrist = vector(0.39, 1.56, 0.33);
    const hand = sphere(actor, 0.13, [0, 0, 0], skin);
    const can = new T.Group();
    actor.add(can);
    const sprayDistance = vector(0.24, 0, 0.28);
    can.rotation.y = Math.atan2(-sprayDistance.x, -sprayDistance.z) - actor.rotation.y;
    cylinder(can, [0.102, 0.102], 0.36, [0, 0.015, 0], material(0xed6253, 0.43));
    cylinder(can, [0.106, 0.106], 0.087, [0, -0.044, 0], cord);
    cylinder(can, [0.106, 0.106], 0.028, [0, -0.165, 0], metal);
    cylinder(can, [0.073, 0.10], 0.044, [0, 0.196, 0], metal);
    box(can, [0.087, 0.052, 0.123], [0, 0.24, 0.033], shoes, 0.021);
    sphere(can, 0.023, [0, 0.24, 0.096], cord, [1, 1, 0.5]);
    const nozzleOffset = vector(0, 0.24, 0.10).applyQuaternion(can.quaternion);
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
      const amount = clamp((progress - start) / (end - start));
      return amount * amount * (3 - 2 * amount);
    };
    const closePosition = cctv.position.clone().addScaledVector(lensDirection, 1.91).add(vector(0, 0.18, 0.80));
    const closeTarget = cctv.position.clone().addScaledVector(lensDirection, 0.19);
    const widePosition = vector(-2.32, 3.36, 5.42);
    const wideTarget = vector(0.13, 2.14, -3.27);
    const look = new T.Vector3();
    const mark = new T.Vector3(), sprayOrigin = new T.Vector3(), gaze = new T.Vector3();
    const hip = new T.Vector3(), ankle = new T.Vector3();

    renderScene = function () {
      const wide = ease(0, 0.43);
      camera.position.copy(closePosition).lerp(widePosition, wide);
      camera.lookAt(look.copy(closeTarget).lerp(wideTarget, wide));

      const first = ease(0.37, 0.55), second = ease(0.571, 0.68);
      reveal(paint, first);
      reveal(accent, second);
      if (progress <= 0.55) paint.curve.getPointAt(first, mark);
      else if (progress >= 0.571) accent.curve.getPointAt(second, mark);
      else mark.copy(paintEnd).lerp(accentStart, ease(0.55, 0.571));
      mark.add(graffiti.position);

      const reach = ease(0.28, 0.37) * (1 - ease(0.68, 0.76));
      const stepping = progress <= 0.55 ? first : second;
      const stride = 0.105 * Math.sin(7 * Math.PI * stepping) * reach;
      actor.position.x = mark.x + 0.92 + 0.23 * ease(0.68, 0.77);
      for (const { side, leg, shoe, sole } of legs) {
        const lift = Math.max(0, side * stride) * 0.42;
        ankle.set(side * 0.23, 0.15 + lift, side * stride);
        hip.set(side * 0.185, 1.06, 0);
        pose(leg, hip, ankle);
        shoe.position.set(ankle.x, 0.113 + lift, ankle.z + 0.089);
        sole.position.set(ankle.x, 0.035 + lift, ankle.z + 0.089);
      }
      sprayOrigin.copy(mark).add(sprayDistance);
      actor.worldToLocal(wrist.copy(sprayOrigin)).sub(nozzleOffset);
      wrist.lerp(restingWrist, 1 - reach);
      elbow.set(0.38 + reach * 0.06, 1.68 + reach * (0.28 + (mark.y - 2.62) * 0.43), 0.12 + reach * 0.27);
      pose(upper, shoulder, elbow);
      pose(forearm, elbow, wrist);
      hand.position.copy(wrist);
      hand.position.x += 0.092;
      can.position.copy(wrist);
      actor.worldToLocal(gaze.copy(mark).lerp(cctv.position, ease(0.71, 0.81)));
      gaze.sub(head.position);
      head.rotation.y = Math.max(-0.54, Math.min(0.55, Math.atan2(gaze.x, gaze.z)));
      head.rotation.x = Math.max(-0.25, Math.min(0.25, -gaze.y * 0.31));

      const spraying = (first > 0 && first < 1) || (second > 0 && second < 1);
      mist.forEach((particle, index) => {
        particle.visible = spraying;
        if (!spraying) return;
        const distance = ((progress * 87 + index * 0.163) % 1) * 0.82 + 0.09;
        particle.position.copy(sprayOrigin).lerp(mark, distance);
        particle.position.x += Math.sin(index * 2.4) * distance * 0.062;
        particle.position.y += Math.cos(index * 2.4) * distance * 0.069;
      });
      bracket.opacity = ease(0.59, 0.72);
      detection.visible = bracket.opacity > 0;
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
