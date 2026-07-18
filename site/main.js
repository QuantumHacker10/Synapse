/* Synapse — interactions : canvas bioluminescent, révélations, nav mobile */
(function () {
  "use strict";

  /* ---------- Révélation au scroll ---------- */
  const reveals = document.querySelectorAll(".reveal");
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          e.target.classList.add("visible");
          io.unobserve(e.target);
        }
      }
    },
    { threshold: 0.15, rootMargin: "0px 0px -5% 0px" }
  );
  reveals.forEach((el) => io.observe(el));

  /* ---------- Header : bordure au scroll ---------- */
  const header = document.querySelector(".site-header");
  const onScroll = () => header.classList.toggle("scrolled", window.scrollY > 24);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ---------- Nav mobile ---------- */
  const toggle = document.querySelector(".nav-toggle");
  const menu = document.getElementById("nav-menu");
  toggle.addEventListener("click", () => {
    const open = menu.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(open));
    toggle.setAttribute("aria-label", open ? "Fermer le menu" : "Ouvrir le menu");
  });
  menu.addEventListener("click", (e) => {
    if (e.target.tagName === "A") {
      menu.classList.remove("open");
      toggle.setAttribute("aria-expanded", "false");
    }
  });

  /* ---------- Canvas bioluminescent du hero ----------
     Un banc de "neurones" flottants reliés par des synapses éphémères :
     évocation directe du concept central (formes apprises par réseaux). */
  const canvas = document.getElementById("bio-canvas");
  const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (canvas && !prefersReduced) {
    const ctx = canvas.getContext("2d");
    let width = 0;
    let height = 0;
    let dpr = 1;
    let particles = [];
    let raf = 0;
    let pointer = { x: -9999, y: -9999 };

    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const target = Math.min(90, Math.floor((width * height) / 16000));
      particles = Array.from({ length: target }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.25,
        vy: (Math.random() - 0.5) * 0.25,
        r: 0.8 + Math.random() * 1.8,
        phase: Math.random() * Math.PI * 2,
        speed: 0.4 + Math.random() * 0.8,
      }));
    }

    const LINK_DIST = 110;

    function step(t) {
      ctx.clearRect(0, 0, width, height);
      const time = t * 0.001;

      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;

        // Légère répulsion autour du pointeur : le banc s'écarte
        const dxp = p.x - pointer.x;
        const dyp = p.y - pointer.y;
        const dp2 = dxp * dxp + dyp * dyp;
        if (dp2 < 120 * 120 && dp2 > 0.01) {
          const d = Math.sqrt(dp2);
          p.x += (dxp / d) * 0.6;
          p.y += (dyp / d) * 0.6;
        }

        if (p.x < -20) p.x = width + 20;
        if (p.x > width + 20) p.x = -20;
        if (p.y < -20) p.y = height + 20;
        if (p.y > height + 20) p.y = -20;
      }

      // Synapses
      ctx.lineWidth = 1;
      for (let i = 0; i < particles.length; i++) {
        const a = particles[i];
        for (let j = i + 1; j < particles.length; j++) {
          const b = particles[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < LINK_DIST * LINK_DIST) {
            const alpha = (1 - Math.sqrt(d2) / LINK_DIST) * 0.16;
            ctx.strokeStyle = "rgba(69, 224, 184, " + alpha.toFixed(3) + ")";
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }

      // Neurones (pulsation douce)
      for (const p of particles) {
        const pulse = 0.45 + 0.55 * (0.5 + 0.5 * Math.sin(time * p.speed + p.phase));
        const r = p.r * (0.8 + 0.5 * pulse);

        const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r * 5);
        grad.addColorStop(0, "rgba(69, 224, 184, " + (0.5 * pulse).toFixed(3) + ")");
        grad.addColorStop(1, "rgba(69, 224, 184, 0)");
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r * 5, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = "rgba(190, 255, 236, " + (0.35 + 0.5 * pulse).toFixed(3) + ")";
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fill();
      }

      raf = requestAnimationFrame(step);
    }

    function start() {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(step);
    }

    window.addEventListener("resize", () => {
      resize();
    });
    canvas.parentElement.addEventListener("pointermove", (e) => {
      const rect = canvas.getBoundingClientRect();
      pointer.x = e.clientX - rect.left;
      pointer.y = e.clientY - rect.top;
    });
    canvas.parentElement.addEventListener("pointerleave", () => {
      pointer.x = -9999;
      pointer.y = -9999;
    });

    // Pause hors écran : économise batterie et CPU
    const heroObserver = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) start();
        else cancelAnimationFrame(raf);
      }
    });
    heroObserver.observe(canvas);

    resize();
  }
})();
