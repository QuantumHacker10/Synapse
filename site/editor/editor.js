(function () {
  const meta = JSON.parse(document.getElementById('synapse-scene').textContent);
  const canvas = document.getElementById('viewport');
  const hierarchy = document.getElementById('hierarchy');
  const inspector = document.getElementById('inspector');
  const lawLabel = document.getElementById('law-label');
  const entityLabel = document.getElementById('entity-label');
  const exportBtn = document.getElementById('export-btn');

  lawLabel.textContent = 'Loi : ' + (meta.activeLawId || '—');

  let device, context, format, pipeline;
  let scene = { nodes: [], buffers: [] };
  let selectedIndex = 0;
  let dragging = false;
  let lastX = 0, lastY = 0;

  async function loadGltf(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error('glTF load failed: ' + url);
    const gltf = await res.json();
    const bin = await loadBuffer(gltf.buffers[0]);
    scene.nodes = gltf.nodes.map((n, i) => ({
      name: n.name || ('Node_' + i),
      translation: n.translation ? [...n.translation] : [0, 0, 0],
      meshIndex: n.mesh,
      extras: n.extras || {},
      primitives: n.mesh != null ? buildPrimitive(gltf, n.mesh, bin) : null
    }));
    entityLabel.textContent = scene.nodes.length + ' entité(s)';
    renderHierarchy();
    renderInspector();
  }

  async function loadBuffer(bufferDesc) {
    if (bufferDesc.uri.startsWith('data:')) {
      const b64 = bufferDesc.uri.split(',')[1];
      const raw = atob(b64);
      const arr = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i);
      return arr.buffer;
    }
    const res = await fetch(bufferDesc.uri);
    return res.arrayBuffer();
  }

  function buildPrimitive(gltf, meshIndex, bin) {
    const prim = gltf.meshes[meshIndex].primitives[0];
    const posAcc = gltf.accessors[prim.attributes.POSITION];
    const idxAcc = gltf.accessors[prim.indices];
    const posView = gltf.bufferViews[posAcc.bufferView];
    const idxView = gltf.bufferViews[idxAcc.bufferView];
    const positions = new Float32Array(bin, posView.byteOffset, posAcc.count * 3);
    const indices = idxAcc.componentType === 5123
      ? new Uint16Array(bin, idxView.byteOffset, idxAcc.count)
      : new Uint32Array(bin, idxView.byteOffset, idxAcc.count);
    return { positions, indices };
  }

  function renderHierarchy() {
    hierarchy.innerHTML = '<h3>Hiérarchie</h3>';
    scene.nodes.forEach((n, i) => {
      const div = document.createElement('div');
      div.className = 'node-item' + (i === selectedIndex ? ' selected' : '');
      div.textContent = n.name;
      div.onclick = () => { selectedIndex = i; renderHierarchy(); renderInspector(); draw(); };
      hierarchy.appendChild(div);
    });
  }

  function renderInspector() {
    const n = scene.nodes[selectedIndex];
    if (!n) { inspector.innerHTML = '<h3>Inspecteur</h3><p>Aucune sélection</p>'; return; }
    inspector.innerHTML = '<h3>Inspecteur</h3>' +
      '<p><strong>' + n.name + '</strong></p>' +
      '<p>Type : ' + (n.extras.synapseType || 'Entity') + '</p>' +
      field('X', n.translation[0], 0) +
      field('Y', n.translation[1], 1) +
      field('Z', n.translation[2], 2) +
      '<p id="status">Cliquez-glissez le viewport pour déplacer l\'entité.</p>';
    inspector.querySelectorAll('input').forEach(inp => {
      inp.onchange = () => {
        const axis = parseInt(inp.dataset.axis, 10);
        n.translation[axis] = parseFloat(inp.value) || 0;
        draw();
      };
    });
  }

  function field(label, value, axis) {
    return '<div class="inspector-field"><label>' + label + '</label>' +
      '<input type="number" step="0.1" data-axis="' + axis + '" value="' + value.toFixed(2) + '" /></div>';
  }

  async function initWebGpu() {
    if (!navigator.gpu) {
      hierarchy.innerHTML += '<p style="color:#f85149">WebGPU requis (Chrome/Edge récent).</p>';
      return;
    }
    const adapter = await navigator.gpu.requestAdapter();
    device = await adapter.requestDevice();
    context = canvas.getContext('webgpu');
    format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });

    const shader = device.createShaderModule({
      code: `
        struct VSOut { @builtin(position) pos: vec4f, @location(0) col: vec3f };
        @vertex fn vs(@location(0) p: vec3f, @location(1) c: vec3f) -> VSOut {
          var o: VSOut; o.pos = vec4f(p, 1.0); o.col = c; return o;
        }
        @fragment fn fs(in: VSOut) -> @location(0) vec4f { return vec4f(in.col, 1.0); }
      `
    });
    pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: shader, entryPoint: 'vs',
        buffers: [{ arrayStride: 24, attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' },
          { shaderLocation: 1, offset: 12, format: 'float32x3' }
        ]}]
      },
      fragment: { module: shader, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' }
    });
  }

  function nodeColor(i) {
    return i === selectedIndex ? [0.27, 0.88, 0.72] : [0.35, 0.55, 0.85];
  }

  function draw() {
    if (!device) return;
    resizeCanvas();
    const encoder = device.createCommandEncoder();
    const view = context.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view, clearValue: { r: 0.04, g: 0.06, b: 0.1, a: 1 }, loadOp: 'clear', storeOp: 'store' }]
    });
    pass.setPipeline(pipeline);
    scene.nodes.forEach((n, i) => {
      if (!n.primitives) return;
      const verts = transformVerts(n.primitives.positions, n.translation);
      const colors = new Float32Array(verts.length);
      const c = nodeColor(i);
      for (let v = 0; v < verts.length / 3; v++) { colors[v * 3] = c[0]; colors[v * 3 + 1] = c[1]; colors[v * 3 + 2] = c[2]; }
      const interleaved = interleave(verts, colors);
      const vb = device.createBuffer({ size: interleaved.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(vb, 0, interleaved);
      const ib = device.createBuffer({ size: n.primitives.indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(ib, 0, n.primitives.indices);
      pass.setVertexBuffer(0, vb);
      pass.setIndexBuffer(ib, n.primitives.indices instanceof Uint16Array ? 'uint16' : 'uint32');
      pass.drawIndexed(n.primitives.indices.length);
    });
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  function transformVerts(positions, t) {
    const out = new Float32Array(positions.length);
    for (let i = 0; i < positions.length; i += 3) {
      out[i] = positions[i] + t[0];
      out[i + 1] = positions[i + 1] + t[1];
      out[i + 2] = positions[i + 2] + t[2];
    }
    return out;
  }

  function interleave(pos, col) {
    const n = pos.length / 3;
    const out = new Float32Array(n * 6);
    for (let i = 0; i < n; i++) {
      out[i * 6] = pos[i * 3]; out[i * 6 + 1] = pos[i * 3 + 1]; out[i * 6 + 2] = pos[i * 3 + 2];
      out[i * 6 + 3] = col[i * 3]; out[i * 6 + 4] = col[i * 3 + 1]; out[i * 6 + 5] = col[i * 3 + 2];
    }
    return out;
  }

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }

  canvas.addEventListener('pointerdown', e => { dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.classList.add('dragging'); });
  canvas.addEventListener('pointerup', () => { dragging = false; canvas.classList.remove('dragging'); });
  canvas.addEventListener('pointermove', e => {
    if (!dragging) return;
    const n = scene.nodes[selectedIndex];
    if (!n) return;
    n.translation[0] += (e.clientX - lastX) * 0.01;
    n.translation[2] += (e.clientY - lastY) * 0.01;
    lastX = e.clientX; lastY = e.clientY;
    renderInspector();
    draw();
  });

  exportBtn.onclick = () => {
    const gltf = {
      asset: { version: '2.0', generator: 'Synapse Web Editor export' },
      scene: 0,
      scenes: [{ name: meta.sceneName, nodes: scene.nodes.map((_, i) => i) }],
      nodes: scene.nodes.map(n => ({ name: n.name, translation: n.translation, extras: n.extras }))
    };
    const blob = new Blob([JSON.stringify(gltf, null, 2)], { type: 'model/gltf+json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'scene-edited.gltf';
    a.click();
  };

  async function boot() {
    await initWebGpu();
    await loadGltf(meta.gltfUrl || 'demo.gltf');
    draw();
    requestAnimationFrame(function loop() { draw(); requestAnimationFrame(loop); });
  }

  boot().catch(err => { hierarchy.innerHTML += '<p style="color:#f85149">' + err.message + '</p>'; });
})();
