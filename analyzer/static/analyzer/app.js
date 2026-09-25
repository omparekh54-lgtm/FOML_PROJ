(() => {
  "use strict";

  const MAX_BYTES = 10 * 1024 * 1024;
  const ACCEPTED = ["image/jpeg", "image/png", "image/bmp", "image/tiff"];
  const REGIONS = [
    "upper-left", "upper-center", "upper-right",
    "mid-left", "center", "mid-right",
    "lower-left", "lower-center", "lower-right",
  ];
  const ABNORMAL_NAME = { chest: "Signs of pneumonia", bone: "Signs of a fracture" };
  const LOADING_MESSAGES = [
    "Standardising contrast and size…",
    "Projecting onto principal components…",
    "Routing to the right body-part model…",
    "Working out what drove the prediction…",
  ];

  const $ = (id) => document.getElementById(id);
  const input = $("file-input");
  const dropzone = $("dropzone");
  const preview = $("dz-preview");
  const dzEmpty = $("dz-empty");
  const dzActions = $("dz-actions");
  const fileName = $("file-name");
  const emptyState = $("empty-state");
  const loading = $("loading");
  const loadingText = $("loading-text");
  const result = $("result");
  const csrf = document.querySelector('meta[name="csrf-token"]').content;

  let previewUrl = null;
  let loadingTimer = null;
  let requestId = 0;

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
  const pct = (x) => `${Math.round(x * 100)}%`;
  const cap = (s) => s ? s[0].toUpperCase() + s.slice(1) : s;

  // ---------- input handling ----------
  input.addEventListener("change", () => {
    if (input.files && input.files[0]) handleFile(input.files[0]);
  });

  ["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  }));
  ["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  }));
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
  dropzone.addEventListener("click", (e) => {
    if (dropzone.classList.contains("has-image")) e.preventDefault();
  });

  document.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) handleFile(item.getAsFile(), "Pasted image");
  });

  $("reset-btn").addEventListener("click", reset);

  document.querySelectorAll(".sample").forEach((btn) => {
    btn.addEventListener("click", async () => {
      document.querySelectorAll(".sample").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      try {
        const blob = await (await fetch(btn.dataset.src)).blob();
        const file = new File([blob], btn.dataset.src.split("/").pop(), { type: blob.type || "image/jpeg" });
        handleFile(file, `Sample · ${btn.dataset.label}`);
      } catch {
        showError("Couldn't load that sample. Please try again.");
      }
    });
  });

  function reset() {
    requestId++;
    input.value = "";
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    preview.hidden = true;
    preview.removeAttribute("src");
    dzEmpty.hidden = false;
    dzActions.hidden = true;
    dropzone.classList.remove("has-image");
    document.querySelectorAll(".sample").forEach((b) => b.classList.remove("active"));
    stopLoading();
    result.hidden = true;
    result.innerHTML = "";
    emptyState.hidden = false;
  }

  function handleFile(file, label) {
    if (!file) return;
    if (file.type && !ACCEPTED.includes(file.type)) {
      showError("Please choose a JPG, PNG, BMP or TIFF image.");
      return;
    }
    if (file.size > MAX_BYTES) {
      showError("That file is larger than 10 MB.");
      return;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.hidden = false;
    dzEmpty.hidden = true;
    dropzone.classList.add("has-image");
    dzActions.hidden = false;
    fileName.textContent = label || file.name;
    analyze(file);
  }

  // ---------- request ----------
  async function analyze(file) {
    const id = ++requestId;
    startLoading();
    const body = new FormData();
    body.append("image", file);
    let data;
    try {
      const res = await fetch("/api/analyze/", {
        method: "POST",
        body,
        headers: { "X-CSRFToken": csrf },
        credentials: "same-origin",
      });
      data = await res.json().catch(() => ({ status: "error", message: `Server error (${res.status}). Please try again.` }));
    } catch {
      data = { status: "error", message: "Couldn't reach the server. Check your connection and try again." };
    }
    if (id !== requestId) return; // a newer upload replaced this one
    stopLoading();
    render(data);
    if (window.innerWidth < 960) $("result-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function startLoading() {
    emptyState.hidden = true;
    result.hidden = true;
    loading.hidden = false;
    let i = 0;
    loadingText.textContent = LOADING_MESSAGES[0];
    clearInterval(loadingTimer);
    loadingTimer = setInterval(() => {
      i = (i + 1) % LOADING_MESSAGES.length;
      loadingText.textContent = LOADING_MESSAGES[i];
    }, 900);
  }

  function stopLoading() {
    clearInterval(loadingTimer);
    loading.hidden = true;
  }

  function showError(message) {
    stopLoading();
    render({ status: "error", message });
  }

  // ---------- rendering ----------
  const ICONS = {
    check: '<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
    alert: '<svg viewBox="0 0 24 24"><path d="M12 8v5m0 3.5v.01"/></svg>',
    stop: '<svg viewBox="0 0 24 24"><path d="M7 7l10 10M17 7L7 17"/></svg>',
    warn: '<svg viewBox="0 0 24 24"><path d="M12 3l9.5 17h-19L12 3z"/><path d="M12 10v4m0 3v.01"/></svg>',
  };

  function render(d) {
    emptyState.hidden = true;
    result.hidden = false;
    result.className = "fade-in";
    if (d.status !== "ok") {
      result.innerHTML = renderReject(d);
      return;
    }
    result.innerHTML = renderOk(d);
    wireHighlights(d);
  }

  function renderReject(d) {
    const titles = {
      not_xray: "This doesn't look like an X-ray",
      unsupported_body_part: "Body part not supported yet",
      models_not_trained: "Models aren't ready",
      error: "Something went wrong",
    };
    const img = d.processed_image_png
      ? `<img src="data:image/png;base64,${d.processed_image_png}" alt="The image as the model saw it">`
      : "";
    return `
      <div class="reject">
        <div class="verdict-icon">${ICONS.stop}</div>
        <h2>${esc(titles[d.status] || titles.error)}</h2>
        <p>${esc(d.message || "Please try a different image.")}</p>
        ${img}
      </div>`;
  }

  function renderOk(d) {
    const isNormal = d.abnormality_label === "normal";
    const cls = isNormal ? "normal" : "abnormal";
    const headline = isNormal ? "Looks normal" : (ABNORMAL_NAME[d.body_part] || "Possible abnormality");
    const part = cap(d.body_part);

    const warnings = (d.warnings || []).map((w) =>
      `<div class="callout">${ICONS.warn}<span>${esc(w)}</span></div>`).join("");

    const maxAbs = Math.max(...d.contributions.map((c) => Math.abs(c.contribution)), 1e-9);
    const contribRows = d.contributions.map((c, i) => {
      const width = (Math.abs(c.contribution) / maxAbs) * 50;
      return `
        <div class="contrib-row" tabindex="0" data-region="${esc(c.region)}" data-i="${i}"
             aria-label="Component ${c.component}, ${esc(c.region)}, pushed toward ${esc(c.toward)}, ${pct(c.share)} of total influence">
          <div class="contrib-name"><b>${esc(cap(c.region))}</b><span>component #${c.component} · ${pct(c.share)}</span></div>
          <div class="contrib-track"><div class="contrib-bar toward-${c.toward}" style="width:${width}%"></div></div>
        </div>`;
    }).join("");

    const treeAgrees = d.reasoning_label === d.abnormality_label;
    const ruleItems = (d.explanation_steps || []).map((s) => `
      <li>Pattern strength around the <b>${esc(s.region)}</b> (component #${s.component}) was
        <b>${s.direction}</b> than the rule's split point
        <span class="mono">(${s.value.toFixed(2)} ${s.direction === "higher" ? "&gt;" : "≤"} ${s.threshold.toFixed(2)})</span></li>`).join("");

    return `
      <div class="verdict">
        <div class="chips">
          <span class="chip">Body part <b>${esc(part)}</b> · ${pct(d.body_part_confidence)}</span>
          <span class="chip">PCA <b>${d.pca_components} components</b> · ${pct(d.pca_variance_retained)} variance</span>
        </div>
        <div class="verdict-main ${cls}">
          <div class="verdict-icon">${isNormal ? ICONS.check : ICONS.alert}</div>
          <div class="verdict-body">
            <div class="verdict-label">${esc(headline)}</div>
            <p class="verdict-sub">Model confidence ${pct(d.abnormality_confidence)} for "${esc(d.abnormality_label)}"</p>
            <div class="meter" role="img" aria-label="Confidence ${pct(d.abnormality_confidence)}"><span style="width:${d.abnormality_confidence * 100}%"></span></div>
          </div>
        </div>
        ${warnings}
      </div>

      <div class="block">
        <h3 class="block-title">What the model sees</h3>
        <p class="block-sub">Your image is standardised, then described using only ${d.pca_components} principal components. The rebuilt version is literally all the classifier has to go on.</p>
        <div class="views">
          <div class="view"><figure>
            <div class="frame"><img src="${previewUrl}" alt="Your upload"></div>
            <figcaption><b>Your upload</b>Original image</figcaption>
          </figure></div>
          <div class="view"><figure>
            <div class="frame">
              <img class="pixel" src="data:image/png;base64,${d.processed_image_png}" alt="Standardised 128 by 128 model input">
              <div class="grid-overlay" id="grid-overlay">${REGIONS.map((r) => `<div data-region="${r}"></div>`).join("")}</div>
            </div>
            <figcaption><b>Model input</b>128×128, equalised · shaded regions drove the result</figcaption>
          </figure></div>
          <div class="view"><figure>
            <div class="frame"><img class="pixel" src="data:image/png;base64,${d.reconstruction_png}" alt="Image rebuilt from principal components"></div>
            <figcaption><b>PCA reconstruction</b>Rebuilt from ${d.pca_components} components</figcaption>
          </figure></div>
        </div>
      </div>

      <div class="block">
        <h3 class="block-title">What drove this prediction</h3>
        <p class="block-sub">The principal components that moved the decision most. This breakdown is exact: it's how the classifier actually combines them. Hover a row to see its region.</p>
        <div class="contrib-axis"><span>← toward normal</span><span>toward abnormal →</span></div>
        <div class="contrib">${contribRows}</div>
        <p class="contrib-summary">${esc(d.explanation)}</p>
      </div>

      <div class="block">
        <h3 class="block-title">Step-by-step reasoning
          <span class="agree ${treeAgrees ? "yes" : "no"}">${treeAgrees ? "agrees with main model" : "differs from main model"}</span></h3>
        <p class="block-sub">A small decision tree trained to imitate the classifier, so its logic can be read as rules.</p>
        <ol class="rules">
          ${ruleItems}
          <li class="conclusion">So the rules conclude: <b>${esc(d.reasoning_label)}</b></li>
        </ol>
      </div>

      <p class="disclaimer">This is the output of an educational model trained on public datasets, not a diagnosis. Anyone concerned about an X-ray should consult a doctor.</p>`;
  }

  function wireHighlights(d) {
    const overlay = $("grid-overlay");
    if (!overlay) return;
    const cells = Object.fromEntries([...overlay.children].map((c) => [c.dataset.region, c]));
    // Shade each region by the direction of its strongest top contribution.
    const seen = new Set();
    d.contributions.forEach((c) => {
      if (seen.has(c.region) || !cells[c.region]) return;
      seen.add(c.region);
      cells[c.region].classList.add(c.toward === "abnormal" ? "hit-abnormal" : "hit-normal");
    });
    result.querySelectorAll(".contrib-row").forEach((row) => {
      const on = () => cells[row.dataset.region]?.classList.add("focus");
      const off = () => cells[row.dataset.region]?.classList.remove("focus");
      row.addEventListener("mouseenter", on);
      row.addEventListener("mouseleave", off);
      row.addEventListener("focus", on);
      row.addEventListener("blur", off);
    });
  }
})();
