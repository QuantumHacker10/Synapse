window.synapseDownload = function (filename, content, mime) {
  const blob = new Blob([content], { type: mime || 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
};

window.synapseWebGpu = (function () {
  let device, context, format, pipeline, ready = false;

  const shader = `
    struct Uniforms { color: vec4f, mvp: mat4x4f }
    @group(0) @binding(0) var<uniform> u: Uniforms;
    struct VSOut { @builtin(position) pos: vec4f }
    @vertex fn vs(@location(0) position: vec3f) -> VSOut {
      var o: VSOut;
      o.pos = u.mvp * vec4f(position, 1.0);
      return o;
    }
    @fragment fn fs() -> @location(0) vec4f { return u.color; }
  `;

  async function init(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return false;
    device = await adapter.requestDevice();
    context = canvas.getContext('webgpu');
    format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });

    const module = device.createShaderModule({ code: shader });
    pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module,
        entryPoint: 'vs',
        buffers: [{
          arrayStride: 12,
          attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }]
        }]
      },
      fragment: {
        module,
        entryPoint: 'fs',
        targets: [{ format }]
      },
      primitive: { topology: 'triangle-list', cullMode: 'back' }
    });
    ready = true;
    return true;
  }

  function perspective(fov, aspect, near, far) {
    const f = 1.0 / Math.tan(fov / 2);
    const nf = 1 / (near - far);
    return new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) * nf, -1,
      0, 0, (2 * far * near) * nf, 0
    ]);
  }

  function multiply(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        o[c * 4 + r] =
          a[0 * 4 + r] * b[c * 4 + 0] +
          a[1 * 4 + r] * b[c * 4 + 1] +
          a[2 * 4 + r] * b[c * 4 + 2] +
          a[3 * 4 + r] * b[c * 4 + 3];
      }
    }
    return o;
  }

  function translation(x, y, z) {
    return new Float32Array([
      1,0,0,0,
      0,1,0,0,
      0,0,1,0,
      x,y,z,1
    ]);
  }

  function scale(x, y, z) {
    return new Float32Array([
      x,0,0,0,
      0,y,0,0,
      0,0,z,0,
      0,0,0,1
    ]);
  }

  function drawScene(entities) {
    if (!ready) return;
    const canvas = context.canvas;
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.03, g: 0.05, b: 0.09, a: 1 },
        loadOp: 'clear',
        storeOp: 'store'
      }]
    });
    pass.setPipeline(pipeline);

    const cube = new Float32Array([
      -0.5,-0.5, 0.5,  0.5,-0.5, 0.5,  0.5, 0.5, 0.5,
      -0.5,-0.5, 0.5,  0.5, 0.5, 0.5, -0.5, 0.5, 0.5,
      -0.5,-0.5,-0.5, -0.5, 0.5,-0.5,  0.5, 0.5,-0.5,
      -0.5,-0.5,-0.5,  0.5, 0.5,-0.5,  0.5,-0.5,-0.5,
       0.5,-0.5,-0.5,  0.5, 0.5,-0.5,  0.5, 0.5, 0.5,
       0.5,-0.5,-0.5,  0.5, 0.5, 0.5,  0.5,-0.5, 0.5,
      -0.5,-0.5,-0.5, -0.5,-0.5, 0.5, -0.5, 0.5, 0.5,
      -0.5,-0.5,-0.5, -0.5, 0.5, 0.5, -0.5, 0.5,-0.5,
      -0.5, 0.5, 0.5,  0.5, 0.5, 0.5,  0.5, 0.5,-0.5,
      -0.5, 0.5, 0.5,  0.5, 0.5,-0.5, -0.5, 0.5,-0.5,
      -0.5,-0.5,-0.5,  0.5,-0.5,-0.5,  0.5,-0.5, 0.5,
      -0.5,-0.5,-0.5,  0.5,-0.5, 0.5, -0.5,-0.5, 0.5
    ]);
    const vbo = device.createBuffer({ size: cube.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(vbo, 0, cube);
    pass.setVertexBuffer(0, vbo);

    const aspect = canvas.width / Math.max(1, canvas.height);
    const proj = perspective(Math.PI / 4, aspect, 0.1, 100);
    const view = translation(0, -1.5, -10);
    const vp = multiply(proj, view);

    for (const e of entities || []) {
      const model = multiply(translation(e.x || 0, e.y || 0, e.z || 0), scale(e.sx || 1, e.sy || 1, e.sz || 1));
      const mvp = multiply(vp, model);
      const color = e.selected
        ? new Float32Array([0.25, 0.55, 1.0, 1])
        : (e.type === 'Agent' ? new Float32Array([0.2, 0.85, 0.55, 1]) : new Float32Array([0.55, 0.62, 0.78, 1]));
      const uniform = new Float32Array(20);
      uniform.set(color, 0);
      uniform.set(mvp, 4);
      const ubo = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(ubo, 0, uniform);
      const bg = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: ubo } }]
      });
      pass.setBindGroup(0, bg);
      pass.draw(36);
    }

    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  return { init, drawScene };
})();
