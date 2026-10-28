import { useCallback, useEffect, useRef, useState } from "react";

const VIEW_CONFIG = [
  {
    key: "palm_open_front",
    title: "Open palm",
    subtitle: "Palm facing the camera",
    number: "01",
  },
  {
    key: "palm_open_back",
    title: "Back of hand",
    subtitle: "Hand open, back facing the camera",
    number: "02",
  },
  {
    key: "palm_closed_front",
    title: "Closed hand · palm side",
    subtitle: "Make a relaxed fist, palm side visible",
    number: "03",
  },
  {
    key: "palm_closed_back",
    title: "Closed hand · back side",
    subtitle: "Make a relaxed fist, knuckles visible",
    number: "04",
  },
];

const DOSHA_INFO = {
  V: {
    name: "Vata",
    sanskrit: "Air · Space",
    tone: "vata",
    note: "Often associated with movement, lightness, and change in Ayurvedic tradition.",
  },
  P: {
    name: "Pitta",
    sanskrit: "Fire · Water",
    tone: "pitta",
    note: "Often associated with warmth, transformation, and focus in Ayurvedic tradition.",
  },
  K: {
    name: "Kapha",
    sanskrit: "Earth · Water",
    tone: "kapha",
    note: "Often associated with steadiness, structure, and grounding in Ayurvedic tradition.",
  },
};

const AUTO = "Auto (from images)";
const DEFAULT_OBSERVATIONS = {
  vein_visibility: AUTO,
  mole: AUTO,
  skin_texture: AUTO,
  skin_color: AUTO,
  palm_color: AUTO,
  hair_thickness: AUTO,
};

const ICONS = {
  leaf: <><path d="M20.7 3.3C12.4 3.1 6.1 5.1 4.1 10.2c-1.2 3.2.1 6.1 2.9 6.8 5.2 1.3 10.7-4.7 13.7-13.7Z" /><path d="M3.5 21c2.7-5.8 6.9-9.5 12.8-12.8" /><path d="M8 13c.1 2.6 1.4 4.5 3.7 5.2" /></>,
  hand: <><path d="M8 12V5.5a1.6 1.6 0 0 1 3.2 0V11" /><path d="M11.2 10V4.6a1.6 1.6 0 0 1 3.2 0V11" /><path d="M14.4 10.6V6a1.6 1.6 0 0 1 3.2 0v7.5c0 4.3-2.4 7-6.6 7h-.6c-2.4 0-3.8-1.2-5-3.4L3.7 13a1.7 1.7 0 0 1 2.8-1.9L8 13" /></>,
  file: <><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9Z" /><path d="M13 2v7h7M8 13h8M8 17h8" /></>,
  shield: <><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z" /><path d="m9 12 2 2 4-4" /></>,
  arrow: <><path d="M5 12h14M13 6l6 6-6 6" /></>,
  check: <path d="m5 12 4 4L19 6" />,
  cloud: <><path d="M20 16.5a4.5 4.5 0 0 0-1.1-8.9A6 6 0 0 0 7.4 6.3a4.5 4.5 0 0 0 .1 9h12.5Z" /><path d="m9 14 3-3 3 3M12 11v8" /></>,
  upload: <><path d="M12 16V4m-5 5 5-5 5 5" /><path d="M4 17v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" /></>,
  close: <><path d="m18 6-12 12M6 6l12 12" /></>,
  info: <><circle cx="12" cy="12" r="10" /><path d="M12 11v5m0-8h.01" /></>,
  chevron: <path d="m6 9 6 6 6-6" />,
  lock: <><rect x="4" y="10" width="16" height="11" rx="2" /><path d="M8 10V7a4 4 0 1 1 8 0v3m-4 5v2" /></>,
  user: <><circle cx="12" cy="8" r="4" /><path d="M5 21a7 7 0 0 1 14 0" /></>,
  photo: <><rect x="3" y="3" width="18" height="18" rx="3" /><circle cx="8.5" cy="8.5" r="1.5" /><path d="m21 15-5-5L5 21" /></>,
  chart: <><path d="M3 3v18h18" /><path d="m19 9-5 5-4-4-5 5" /></>,
  database: <><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" /></>,
  refresh: <><path d="M20 7v5h-5M4 17v-5h5" /><path d="M5.6 9A7 7 0 0 1 18 6l2 2M4 16l2 2a7 7 0 0 0 12.4-3" /></>,
  logout: <><path d="M10 17l5-5-5-5M15 12H3" /><path d="M12 3h6a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-6" /></>,
  sparkle: <><path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-1.9-5.8L4 11l6.1-2.2L12 3Z" /><path d="m19 14 1.1 2.9L23 18l-2.9 1.1L19 22l-1.1-2.9L15 18l2.9-1.1L19 14Z" /></>,
  menu: <><path d="M4 7h16M4 12h16M4 17h16" /></>,
  download: <><path d="M12 3v12m-5-5 5 5 5-5" /><path d="M4 17v4h16v-4" /></>,
  clipboard: <><rect x="5" y="4" width="14" height="17" rx="2" /><path d="M9 4V2h6v2m-7 6h8m-8 4h8m-8 4h5" /></>,
  clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
};

function Icon({ name, size = 20, className = "" }) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {ICONS[name] || ICONS.sparkle}
    </svg>
  );
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/+$/, "");

function apiUrl(path) {
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

async function apiRequest(path, options = {}) {
  const response = await fetch(apiUrl(path), options);
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : { error: await response.text() };
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status}).`);
  return body;
}

function jsonOptions(body, extraHeaders = {}) {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json", ...extraHeaders },
    body: JSON.stringify(body),
  };
}

function Brand() {
  return (
    <div className="brand-lockup">
      <span className="brand-symbol"><Icon name="leaf" size={22} /></span>
      <span className="brand-copy">
        <strong>Prakriti</strong>
        <small>AYURVEDA · WELLNESS RESEARCH</small>
      </span>
    </div>
  );
}

function App() {
  const [page, setPage] = useState("assessment");
  const [mobileMenu, setMobileMenu] = useState(false);
  const [server, setServer] = useState(null);
  const [assessment, setAssessment] = useState({
    age: 25,
    gender: "",
    observations: { ...DEFAULT_OBSERVATIONS },
    images: {},
    result: null,
  });
  const [batch, setBatch] = useState({
    file: null,
    autoAppend: true,
    jobId: null,
    job: null,
    busy: false,
    error: "",
  });

  useEffect(() => {
    apiRequest("/api/status")
      .then(setServer)
      .catch(() => setServer({ ok: false }));
  }, []);

  const navigation = [
    { id: "assessment", label: "Prakriti assessment", icon: "hand" },
    { id: "batch", label: "WhatsApp batch import", icon: "file" },
    { id: "doctor", label: "Clinician portal", icon: "shield" },
  ];

  const changePage = (next) => {
    setPage(next);
    setMobileMenu(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-inner">
          <button className="brand-button" onClick={() => changePage("assessment")} aria-label="Go to Prakriti assessment">
            <Brand />
          </button>
          <button
            className="mobile-menu-button"
            aria-label={mobileMenu ? "Close navigation" : "Open navigation"}
            aria-expanded={mobileMenu}
            onClick={() => setMobileMenu((open) => !open)}
          >
            <Icon name={mobileMenu ? "close" : "menu"} />
          </button>
          <nav className={`main-navigation ${mobileMenu ? "is-open" : ""}`} aria-label="Main navigation">
            {navigation.map((item) => (
              <button
                key={item.id}
                className={`nav-link ${page === item.id ? "active" : ""}`}
                onClick={() => changePage(item.id)}
                aria-current={page === item.id ? "page" : undefined}
              >
                <Icon name={item.icon} size={17} />
                <span>{item.label}</span>
              </button>
            ))}
          </nav>
          <div className={`service-status ${server?.ok ? "ready" : server ? "offline" : "checking"}`}>
            <span className="status-dot" />
            <span>{server?.ok ? "Service online" : server ? "Service offline" : "Connecting…"}</span>
          </div>
        </div>
      </header>

      <main className="page-content">
        {page === "assessment" && (
          <AssessmentPage value={assessment} onChange={setAssessment} server={server} />
        )}
        {page === "batch" && (
          <BatchImportPage value={batch} onChange={setBatch} />
        )}
        {page === "doctor" && <DoctorPortalPage />}
      </main>

      <footer className="site-footer">
        <div className="footer-inner">
          <span>Prakriti · Ayurvedic wellness research</span>
          <span>Educational use only · Not a medical diagnosis</span>
        </div>
      </footer>
    </div>
  );
}

function PageIntro({ eyebrow, title, description, icon }) {
  return (
    <section className={`page-intro ${icon ? "has-art" : ""}`}>
      <div className="intro-copy">
        <div className="eyebrow"><span className="eyebrow-mark" />{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {icon && <div className={`intro-art ${icon}`} aria-hidden="true">
        <div className="art-halo" />
        <div className="art-leaf leaf-one"><Icon name="leaf" size={70} /></div>
        <div className="art-leaf leaf-two"><Icon name="leaf" size={47} /></div>
        <div className="art-note"><span className="art-note-icon"><Icon name="sparkle" size={18} /></span><span><strong>Rooted in tradition</strong><small>Designed for reflection</small></span></div>
      </div>}
    </section>
  );
}

function SectionHeading({ number, title, description, compact = false }) {
  return (
    <div className={`section-heading ${compact ? "compact" : ""}`}>
      {number && <span className="section-number">{number}</span>}
      <div>
        <h2>{title}</h2>
        {description && <p>{description}</p>}
      </div>
    </div>
  );
}

function InfoNotice({ children, onClose }) {
  return (
    <aside className="notice-card" aria-label="Important information">
      <span className="notice-icon"><Icon name="info" size={19} /></span>
      <div className="notice-copy">{children}</div>
      {onClose && <button className="icon-button notice-close" onClick={onClose} aria-label="Dismiss notice"><Icon name="close" size={17} /></button>}
    </aside>
  );
}

function AssessmentPage({ value, onChange, server }) {
  const [disclaimerOpen, setDisclaimerOpen] = useState(
    () => window.sessionStorage.getItem("prakriti-disclaimer-dismissed") !== "true",
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const imageCount = VIEW_CONFIG.filter(({ key }) => value.images[key]).length;

  const update = (changes) => onChange((previous) => ({ ...previous, ...changes }));
  const updateObservation = (key, selection) => {
    update({ observations: { ...value.observations, [key]: selection }, result: null });
  };

  const uploadImage = async (key, file) => {
    if (!file) return;
    if (!/^image\/(jpeg|png)$/.test(file.type)) {
      setError("Please choose a JPG or PNG image for each hand view.");
      return;
    }
    setError("");
    const reader = new FileReader();
    reader.onload = async () => {
      const image = { dataUrl: reader.result, name: file.name, quality: null, checking: true };
      onChange((previous) => ({
        ...previous,
        images: { ...previous.images, [key]: image },
        result: null,
      }));
      try {
        const quality = await apiRequest("/api/validate-image", jsonOptions({ image: reader.result }));
        onChange((previous) => ({
          ...previous,
          images: previous.images[key]?.dataUrl === reader.result
            ? { ...previous.images, [key]: { ...previous.images[key], quality, checking: false } }
            : previous.images,
        }));
      } catch (validationError) {
        onChange((previous) => ({
          ...previous,
          images: previous.images[key]?.dataUrl === reader.result
            ? { ...previous.images, [key]: { ...previous.images[key], checking: false } }
            : previous.images,
        }));
      }
    };
    reader.onerror = () => setError("That photo could not be opened. Please try another image.");
    reader.readAsDataURL(file);
  };

  const removeImage = (key) => {
    const nextImages = { ...value.images };
    delete nextImages[key];
    update({ images: nextImages, result: null });
  };

  const analyze = async (event) => {
    event.preventDefault();
    if (!server?.ok) {
      setError("The local analysis service is not available. Restart the app and try again.");
      return;
    }
    if (!value.gender) {
      setError("Select your gender before continuing.");
      return;
    }
    if (imageCount !== 4) {
      setError("Add all four hand photos to start your estimate.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const result = await apiRequest("/api/analyze", jsonOptions({
        age: value.age,
        gender: value.gender,
        observations: value.observations,
        images: Object.fromEntries(Object.entries(value.images).map(([key, item]) => [key, item.dataUrl])),
      }));
      update({ result });
      window.setTimeout(() => document.getElementById("analysis-result")?.scrollIntoView({ behavior: "smooth", block: "start" }), 80);
    } catch (analysisError) {
      setError(analysisError.message);
    } finally {
      setBusy(false);
    }
  };

  const startAgain = () => {
    onChange({ age: 25, gender: "", observations: { ...DEFAULT_OBSERVATIONS }, images: {}, result: null });
    setError("");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <>
      <PageIntro
        eyebrow="A small step toward self-discovery"
        title={<>Explore your <em>Prakriti.</em></>}
        description="Share a few details and four clear hand views for an educational estimate inspired by Ayurveda."
        icon="wellness-art"
      />

      <div className="assessment-steps" aria-label="Assessment steps">
        <div className="assessment-step"><span>01</span><div><strong>Your details</strong><small>Age and gender</small></div></div>
        <div className="step-connector" />
        <div className={`assessment-step ${imageCount === 4 ? "complete" : ""}`}><span>{imageCount === 4 ? <Icon name="check" size={17} /> : "02"}</span><div><strong>Hand photos</strong><small>{imageCount} of 4 added</small></div></div>
        <div className="step-connector" />
        <div className={`assessment-step ${value.result ? "complete" : ""}`}><span>{value.result ? <Icon name="check" size={17} /> : "03"}</span><div><strong>Your estimate</strong><small>Review your results</small></div></div>
      </div>

      {disclaimerOpen && (
        <InfoNotice onClose={() => { window.sessionStorage.setItem("prakriti-disclaimer-dismissed", "true"); setDisclaimerOpen(false); }}>
          <strong>An educational research tool</strong>
          <p>This estimate is not a diagnosis or medical advice. Results may not be clinically accurate. Speak with a qualified Ayurvedic practitioner before making health decisions. Your four photos are used for analysis and saved to the project’s connected storage, as in the existing workflow.</p>
        </InfoNotice>
      )}

      <form onSubmit={analyze}>
        <section className="content-card profile-card">
          <SectionHeading number="01" title="A little about you" description="These details help the model interpret your photos." />
          <div className="profile-fields">
            <label className="field">
              <span>Age</span>
              <div className="input-with-unit"><input type="number" min="1" max="120" value={value.age} onChange={(event) => update({ age: event.target.value, result: null })} required /><span>years</span></div>
            </label>
            <label className="field">
              <span>Gender</span>
              <select value={value.gender} onChange={(event) => update({ gender: event.target.value, result: null })} required>
                <option value="" disabled>Select an option</option>
                <option value="female">Female</option>
                <option value="male">Male</option>
              </select>
            </label>
          </div>
        </section>

        <section className="upload-section">
          <SectionHeading number="02" title="Add four hand photos" description="Use natural light, a plain background, and keep your hand in focus." />
          <div className="photo-grid">
            {VIEW_CONFIG.map((view) => (
              <ImagePicker
                key={view.key}
                view={view}
                image={value.images[view.key]}
                onUpload={(file) => uploadImage(view.key, file)}
                onRemove={() => removeImage(view.key)}
              />
            ))}
          </div>
          <div className="upload-progress">
            <div className="progress-label"><span>{imageCount === 4 ? "All four views are ready" : "Photo progress"}</span><strong>{imageCount} / 4</strong></div>
            <div className="progress-track"><span style={{ width: `${imageCount * 25}%` }} /></div>
            <small>{imageCount === 4 ? "You can start the analysis below." : `Add ${4 - imageCount} more ${4 - imageCount === 1 ? "view" : "views"} to continue.`}</small>
          </div>
        </section>

        <details className="content-card observation-card">
          <summary>
            <span className="summary-icon"><Icon name="clipboard" size={19} /></span>
            <span className="summary-copy"><strong>Optional clinical observations</strong><small>Include practitioner observations if available</small></span>
            <Icon name="chevron" className="summary-chevron" size={18} />
          </summary>
          <div className="observation-content">
            <p className="muted-copy">Leave a field on automatic to use the computer vision estimate from your images.</p>
            <div className="observation-grid">
              <ObservationField label="Vein visibility" value={value.observations.vein_visibility} name="vein_visibility" options={["Low (Deep / Concealed)", "Medium", "High (Prominent / Superficial)"]} onChange={updateObservation} />
              <ObservationField label="Mole on hand or wrist" value={value.observations.mole} name="mole" options={["No", "Yes"]} onChange={updateObservation} />
              <ObservationField label="Skin texture" value={value.observations.skin_texture} name="skin_texture" options={["Normal / Smooth", "Dry", "Rough", "Oily"]} onChange={updateObservation} />
              <ObservationField label="Skin color" value={value.observations.skin_color} name="skin_color" options={["Fair", "Medium", "Dark"]} onChange={updateObservation} />
              <ObservationField label="Palm color" value={value.observations.palm_color} name="palm_color" options={["Pink-Red", "Pink-White", "Brown"]} onChange={updateObservation} />
              <ObservationField label="Hair thickness" value={value.observations.hair_thickness} name="hair_thickness" options={["Thin / Sparse", "Medium", "Thick"]} onChange={updateObservation} />
            </div>
          </div>
        </details>

        {error && <div className="inline-error" role="alert"><Icon name="info" size={18} />{error}</div>}
        <div className="analysis-action">
          <div className="action-note"><Icon name="shield" size={18} /><span>Your results are an educational estimate, not a diagnosis.</span></div>
          <button className="button button-primary button-large" type="submit" disabled={busy || imageCount !== 4 || !value.gender}>
            {busy ? <><span className="spinner" />Analyzing your photos…</> : <>View my estimate <Icon name="arrow" size={18} /></>}
          </button>
        </div>
      </form>

      {value.result && <AnalysisResult result={value.result} onStartAgain={startAgain} />}

      <details className="dosha-guide">
        <summary><span><Icon name="book" size={18} /> A brief guide to the doshas</span><Icon name="chevron" size={18} /></summary>
        <div className="guide-content">
          {Object.entries(DOSHA_INFO).map(([key, dosha]) => (
            <article className={`guide-item ${dosha.tone}`} key={key}>
              <span className="guide-symbol">{key === "V" ? "◌" : key === "P" ? "◉" : "◍"}</span>
              <div><strong>{dosha.name}</strong><small>{dosha.sanskrit}</small><p>{dosha.note}</p></div>
            </article>
          ))}
          <p className="guide-disclaimer">Doshas are traditional Ayurvedic concepts. This tool provides a preliminary model estimate only.</p>
        </div>
      </details>
    </>
  );
}

function ObservationField({ label, name, value, options, onChange }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(name, event.target.value)}>
        <option value={AUTO}>Automatic estimate</option>
        {options.map((option) => <option key={option} value={option}>{option}</option>)}
      </select>
    </label>
  );
}

function ImagePicker({ view, image, onUpload, onRemove }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const acceptFile = (files) => {
    const file = files?.[0];
    if (file) onUpload(file);
  };

  return (
    <article
      className={`photo-card ${image ? "has-image" : ""} ${dragging ? "is-dragging" : ""}`}
      onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
      onDragLeave={(event) => { if (event.currentTarget === event.target) setDragging(false); }}
      onDrop={(event) => { event.preventDefault(); setDragging(false); acceptFile(event.dataTransfer.files); }}
    >
      <input ref={inputRef} type="file" accept="image/jpeg,image/png,.jpg,.jpeg,.png" className="visually-hidden" onChange={(event) => { acceptFile(event.target.files); event.target.value = ""; }} />
      <div className="photo-card-head"><span className="photo-step">{view.number}</span><div><h3>{view.title}</h3><p>{view.subtitle}</p></div>
        {image && <button className="icon-button remove-photo" type="button" onClick={onRemove} aria-label={`Remove ${view.title} photo`}><Icon name="close" size={17} /></button>}
      </div>
      {image ? (
        <div className="photo-preview">
          <img src={image.dataUrl} alt={`${view.title} preview`} />
          <div className="photo-file-label"><Icon name="photo" size={15} /><span title={image.name}>{image.name}</span><Icon name="check" size={16} className="photo-ready-icon" /></div>
          {image.checking && <div className="quality-message"><span className="spinner spinner-small" />Checking photo quality…</div>}
          {!image.checking && image.quality && (
            <div className={`quality-message ${image.quality.valid ? "quality-good" : "quality-warning"}`}>
              <Icon name={image.quality.valid ? "check" : "info"} size={15} />
              <span>{image.quality.valid ? `Looks clear · sharpness ${image.quality.blur_score.toFixed(0)}` : image.quality.issues.join(" · ")}</span>
            </div>
          )}
          <button className="photo-replace" type="button" onClick={() => inputRef.current?.click()}>Replace photo</button>
        </div>
      ) : (
        <button className="photo-dropzone" type="button" onClick={() => inputRef.current?.click()}>
          <span className="dropzone-icon"><Icon name="upload" size={20} /></span>
          <strong>Choose a photo</strong>
          <small>or drag and drop · JPG or PNG</small>
        </button>
      )}
    </article>
  );
}

function AnalysisResult({ result, onStartAgain }) {
  const analysis = result.analysis;
  const save = result.save || {};
  const primaryDosha = DOSHA_INFO[analysis.dominant_class] || DOSHA_INFO.K;
  const probabilities = Object.entries(analysis.probabilities).sort((a, b) => b[1] - a[1]);
  const markers = Object.entries(analysis.effective_markers || {});
  const secondaryVisible = analysis.secondary_class && analysis.secondary_confidence >= 0.25;

  return (
    <section className="analysis-result" id="analysis-result" aria-live="polite">
      <div className="result-section-title"><span className="section-number">03</span><div><span className="eyebrow">Your estimate is ready</span><h2>A reflection of your Prakriti</h2></div></div>
      <article className={`result-hero ${primaryDosha.tone}`}>
        <div className="result-orbit"><span className="result-orbit-core">{primaryDosha.name[0]}</span><i /><i /><i /></div>
        <div className="result-intro">
          <span className="result-kicker">DOMINANT DOSHA ESTIMATE</span>
          <h3>{primaryDosha.name}</h3>
          <p>{primaryDosha.sanskrit}</p>
          <div className="result-confidence"><strong>{(analysis.confidence * 100).toFixed(1)}%</strong><span>model confidence</span></div>
        </div>
        <div className="result-summary">
          <Icon name="sparkle" size={19} />
          <p>{primaryDosha.note}</p>
          {secondaryVisible && <small>Next closest estimate: {DOSHA_INFO[analysis.secondary_class]?.name || analysis.secondary_class} ({(analysis.secondary_confidence * 100).toFixed(0)}%). This is a lower-confidence tendency, not a clinical dual-dosha result.</small>}
        </div>
      </article>

      <div className="result-columns">
        <section className="content-card result-panel">
          <div className="panel-title"><span className="panel-icon"><Icon name="chart" size={18} /></span><div><h3>Probability distribution</h3><p>How the model weighted each dosha</p></div></div>
          <div className="probability-list">
            {probabilities.map(([key, probability]) => {
              const dosha = DOSHA_INFO[key] || { name: key, tone: "kapha" };
              return <div className={`probability-row ${dosha.tone}`} key={key}>
                <div className="probability-name"><span className="probability-dot" /><strong>{dosha.name}</strong><b>{(probability * 100).toFixed(1)}%</b></div>
                <div className="probability-track"><span style={{ width: `${Math.max(0, Math.min(100, probability * 100))}%` }} /></div>
              </div>;
            })}
          </div>
          <small className="model-caption">{analysis.model_name} · Image-based estimate</small>
        </section>

        <section className="content-card result-panel marker-panel">
          <div className="panel-title"><span className="panel-icon"><Icon name="clipboard" size={18} /></span><div><h3>Observed markers</h3><p>From your photos and selected observations</p></div></div>
          <div className="marker-list">
            {markers.map(([label, [value, source]]) => <div className="marker-row" key={label}>
              <span>{label}</span><strong>{value}</strong><small>{source === "Practitioner" ? "Entered" : "Estimated"}</small>
            </div>)}
          </div>
        </section>
      </div>

      <div className={`save-status ${save.record_id ? "saved" : "save-note"}`}>
        <span className="save-status-icon"><Icon name={save.record_id ? "cloud" : "database"} size={18} /></span>
        <div><strong>{save.record_id ? "Analysis saved" : "Analysis complete"}</strong><p>{save.record_id ? `Your analysis was stored. Reference ID: ${save.user_id}` : "The result is ready. Local storage is updated; cloud confirmation may be unavailable."}</p></div>
        {!save.record_id && save.errors?.length > 0 && <span className="save-error-detail" title={save.errors.join("\n")}>Save details</span>}
      </div>

      <div className="result-footer-actions">
        <p>Use this result as a starting point for a conversation with a qualified Ayurvedic practitioner.</p>
        <button className="button button-secondary" type="button" onClick={onStartAgain}><Icon name="refresh" size={17} /> Start a new assessment</button>
      </div>
    </section>
  );
}

function BatchImportPage({ value, onChange }) {
  const [showInstructions, setShowInstructions] = useState(false);
  const job = value.job;

  useEffect(() => {
    if (!value.jobId || job?.state === "complete" || job?.state === "error") return undefined;
    let cancelled = false;
    let timer;
    const poll = async () => {
      try {
        const latest = await apiRequest(`/api/jobs/${value.jobId}`);
        if (cancelled) return;
        onChange((previous) => ({ ...previous, job: latest }));
        if (latest.state === "running") timer = window.setTimeout(poll, 1100);
      } catch (requestError) {
        if (!cancelled) onChange((previous) => ({ ...previous, error: requestError.message }));
      }
    };
    poll();
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [value.jobId, job?.state, onChange]);

  const startImport = async () => {
    if (!value.file) return;
    onChange((previous) => ({ ...previous, busy: true, error: "", job: null, jobId: null }));
    try {
      const response = await fetch(apiUrl("/api/whatsapp/import"), {
        method: "POST",
        headers: { "Content-Type": "application/zip", "X-Auto-Append": String(value.autoAppend), "X-Upload-Filename": encodeURIComponent(value.file.name) },
        body: value.file,
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || "The batch import could not be started.");
      onChange((previous) => ({ ...previous, busy: false, jobId: body.job_id, job: { state: "running", progress: 0, message: "Starting the batch import…" } }));
    } catch (requestError) {
      onChange((previous) => ({ ...previous, busy: false, error: requestError.message }));
    }
  };

  const commitImport = async () => {
    if (!value.jobId) return;
    onChange((previous) => ({ ...previous, busy: true, error: "" }));
    try {
      const result = await apiRequest("/api/whatsapp/commit", jsonOptions({ job_id: value.jobId }));
      onChange((previous) => ({ ...previous, busy: false, job: { ...previous.job, result: { ...previous.job.result, ...result } } }));
    } catch (requestError) {
      onChange((previous) => ({ ...previous, busy: false, error: requestError.message }));
    }
  };

  const downloadCsv = () => {
    if (!job?.result?.rows) return;
    const rows = job.result.rows;
    const columns = [
      ["No", "no"], ["Name", "name"], ["Age", "age"], ["Gender", "gender"],
      ["Skin Color", "skin_color"], ["Skin Texture", "skin_texture"], ["Palm Color", "palm_color"],
      ["Hair", "hair"], ["Veins", "veins"], ["Mole", "mole"], ["Predicted Prakriti", "predicted_dosha"],
      ["Confidence", "confidence"], ["Kapha %", "kapha_prob"], ["Pitta %", "pitta_prob"], ["Vata %", "vata_prob"],
      ["Review Required", "needs_review"], ["Review Reasons", "review_reasons"],
    ];
    const escape = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
    const csv = [columns.map(([label]) => escape(label)).join(","), ...rows.map((row) => columns.map(([, key]) => {
      const value = Array.isArray(row[key]) ? row[key].join("; ") : row[key];
      return escape(value);
    }).join(","))].join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "whatsapp_import_metadata.csv";
    link.click();
    URL.revokeObjectURL(url);
  };

  const clearImport = () => onChange({ file: null, autoAppend: true, jobId: null, job: null, busy: false, error: "" });
  const completed = job?.state === "complete" && job.result;

  return (
    <>
      <PageIntro eyebrow="For research teams" title="WhatsApp batch import" description="Bring a WhatsApp export into the existing image-classification and review workflow." icon="batch-art" />

      <div className="batch-layout">
        <section className="content-card batch-card">
          <SectionHeading number="01" title="Choose an export" description="Upload the ZIP file created with media included." />
          <div className={`zip-drop ${value.file ? "has-file" : ""}`}>
            <input type="file" accept=".zip,application/zip,application/x-zip-compressed" className="visually-hidden" id="whatsapp-zip" onChange={(event) => onChange((previous) => ({ ...previous, file: event.target.files?.[0] || null, job: null, jobId: null, error: "" }))} />
            <label htmlFor="whatsapp-zip" className="zip-drop-label">
              <span className="zip-icon"><Icon name={value.file ? "check" : "upload"} size={23} /></span>
              <strong>{value.file ? value.file.name : "Drop your ZIP file here"}</strong>
              <small>{value.file ? `${(value.file.size / (1024 * 1024)).toFixed(1)} MB · Select another file to replace` : "or browse files · ZIP format"}</small>
              <span className="button button-secondary">{value.file ? "Choose another file" : "Browse files"}</span>
            </label>
          </div>

          <button className="text-button instructions-trigger" type="button" onClick={() => setShowInstructions((open) => !open)} aria-expanded={showInstructions}>
            <Icon name="info" size={17} /> How to export a WhatsApp chat <Icon name="chevron" size={15} className={showInstructions ? "rotated" : ""} />
          </button>
          {showInstructions && <ol className="instruction-list"><li>Open the chat or group on your phone.</li><li>Choose <strong>Export chat</strong> from its menu.</li><li>Select <strong>Include media</strong>, then transfer the ZIP file to this computer.</li></ol>}

          <label className="check-row">
            <input type="checkbox" checked={value.autoAppend} onChange={(event) => onChange((previous) => ({ ...previous, autoAppend: event.target.checked }))} />
            <span className="custom-check"><Icon name="check" size={14} /></span>
            <span><strong>Save imported subjects automatically</strong><small>Append records to the master dataset and sync to Supabase. You can turn this off to review first.</small></span>
          </label>

          {value.error && <div className="inline-error" role="alert"><Icon name="info" size={18} />{value.error}</div>}
          <button className="button button-primary button-large full-width" type="button" onClick={startImport} disabled={!value.file || value.busy || job?.state === "running"}>
            {value.busy && !job?.jobId ? <><span className="spinner" />Preparing import…</> : <>Start batch import <Icon name="arrow" size={18} /></>}
          </button>
          <p className="local-processing-note"><Icon name="shield" size={15} /> The archive is processed by this project’s local analysis pipeline.</p>
        </section>

        <aside className="batch-aside">
          <section className="workflow-card">
            <span className="workflow-icon"><Icon name="database" size={20} /></span>
            <h3>One familiar workflow</h3>
            <p>Chat details and hand photos follow the same classifier, dataset, and clinical review pipeline already used by this project.</p>
            <ul><li><Icon name="check" size={15} /> Extract participant details</li><li><Icon name="check" size={15} /> Classify hand-photo views</li><li><Icon name="check" size={15} /> Flag records for review</li></ul>
          </section>
          <div className="batch-tip"><Icon name="info" size={17} /><p>Use an export that includes both the chat text and its attached images. Missing media can limit the analysis.</p></div>
        </aside>
      </div>

      {job?.state === "running" && <section className="content-card job-progress">
        <div className="job-progress-title"><span className="spinner spinner-dark" /><div><strong>Import in progress</strong><small>{job.message || "Processing the archive…"}</small></div><b>{job.progress || 0}%</b></div>
        <div className="progress-track"><span style={{ width: `${job.progress || 0}%` }} /></div>
      </section>}

      {job?.state === "error" && <div className="inline-error job-error" role="alert"><Icon name="info" size={18} /><span><strong>Import stopped:</strong> {job.error || "Please check the ZIP file and try again."}</span></div>}

      {completed && <section className="batch-results">
        <SectionHeading number="02" title="Import results" description="Review the extracted participant details and classifications." />
        <div className="batch-metrics"><MetricCard label="Subjects in archive" value={job.result.total} icon="user" /><MetricCard label="Saved to dataset" value={job.result.auto_append ? job.result.saved_count : "Pending"} icon="database" /></div>
        <div className="results-toolbar"><div><h3>Participant preview</h3><p>{job.result.total} records returned by the batch pipeline</p></div><button className="button button-secondary" type="button" onClick={downloadCsv}><Icon name="download" size={16} /> Download CSV</button></div>
        <div className="table-scroll"><table className="results-table"><thead><tr><th>No.</th><th>Name</th><th>Age</th><th>Gender</th><th>Skin</th><th>Texture</th><th>Palm</th><th>Hair</th><th>Veins</th><th>Mole</th><th>Prediction</th><th>Confidence</th><th>Dosha split</th><th>Review</th></tr></thead><tbody>
          {job.result.rows.map((row) => <tr key={`${row.no}-${row.name}`}><td>{row.no}</td><td className="table-name">{row.name}</td><td>{row.age}</td><td>{row.gender}</td><td>{row.skin_color}</td><td>{row.skin_texture}</td><td>{row.palm_color}</td><td>{row.hair}</td><td>{row.veins}</td><td>{row.mole}</td><td><span className="dosha-pill">{row.predicted_dosha}</span></td><td>{(row.confidence * 100).toFixed(1)}%</td><td>K {(row.kapha_prob * 100).toFixed(0)} · P {(row.pitta_prob * 100).toFixed(0)} · V {(row.vata_prob * 100).toFixed(0)}</td><td><span className={`review-pill ${row.needs_review ? "review-needed" : "review-clear"}`}>{row.needs_review ? "Review" : "Ready"}</span></td></tr>)}
        </tbody></table></div>
        {!job.result.auto_append && job.result.total > 0 && <div className="commit-row"><div><strong>Ready to add these subjects?</strong><small>Review the records above before appending them to the master dataset.</small></div><button className="button button-primary" type="button" onClick={commitImport} disabled={value.busy}>{value.busy ? <><span className="spinner" />Saving…</> : <>Commit and save <Icon name="arrow" size={17} /></>}</button></div>}
        {job.logs && <details className="log-details"><summary><Icon name="clipboard" size={16} /> View import logs <Icon name="chevron" size={15} /></summary><pre>{job.logs}</pre></details>}
        <button className="text-button new-batch-button" type="button" onClick={clearImport}><Icon name="refresh" size={16} /> Start another import</button>
      </section>}
    </>
  );
}

function MetricCard({ label, value, icon }) {
  return <div className="metric-card"><span className="metric-icon"><Icon name={icon} size={19} /></span><div><small>{label}</small><strong>{value}</strong></div></div>;
}

function DoctorPortalPage() {
  const [token, setToken] = useState(() => window.sessionStorage.getItem("prakriti-doctor-token") || "");
  const [pin, setPin] = useState("");
  const [error, setError] = useState("");
  const [loginBusy, setLoginBusy] = useState(false);
  const [summary, setSummary] = useState(null);
  const [queueMode, setQueueMode] = useState("pending");
  const [selectedId, setSelectedId] = useState("");
  const [subject, setSubject] = useState(null);
  const [subjectLoading, setSubjectLoading] = useState(false);
  const [form, setForm] = useState(null);
  const [saving, setSaving] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");
  const [tab, setTab] = useState("review");
  const [retrainJobId, setRetrainJobId] = useState(() => window.sessionStorage.getItem("prakriti-retrain-job") || "");
  const [retrainJob, setRetrainJob] = useState(null);
  const [trainingBusy, setTrainingBusy] = useState(false);

  const requestWithToken = useCallback((path, options = {}) => apiRequest(path, {
    ...options,
    headers: { ...(options.headers || {}), "X-Doctor-Token": token },
  }), [token]);

  const refreshSummary = useCallback(async () => {
    if (!token) return;
    try {
      const data = await requestWithToken("/api/doctor/summary");
      setSummary(data);
      const active = queueMode === "pending" ? data.pending : data.verified;
      setSelectedId((current) => active.some((entry) => entry.id === current) ? current : active[0]?.id || "");
    } catch (requestError) {
      setError(requestError.message);
      if (requestError.message.includes("sign in")) signOut();
    }
  }, [token, requestWithToken, queueMode]);

  useEffect(() => { refreshSummary(); }, [refreshSummary]);

  useEffect(() => {
    if (!token || !selectedId) { setSubject(null); setForm(null); return undefined; }
    let cancelled = false;
    setSubjectLoading(true);
    requestWithToken(`/api/doctor/subject?id=${encodeURIComponent(selectedId)}`)
      .then((data) => {
        if (cancelled) return;
        setSubject(data);
        setForm({ dosha: data.defaults.dosha, features: {
          vein_visibility: data.defaults.vein,
          mole: data.defaults.mole,
          skin_texture: data.defaults.skin_texture,
          skin_color: data.defaults.skin_color,
          palm_color: data.defaults.palm_color,
          hair_thickness: data.defaults.hair_thickness,
          hair_color: data.defaults.hair_color,
        }, notes: data.defaults.notes === "Awaiting clinical exam" ? "Clinically verified by Doctor" : data.defaults.notes });
        setError("");
      })
      .catch((requestError) => { if (!cancelled) setError(requestError.message); })
      .finally(() => { if (!cancelled) setSubjectLoading(false); });
    return () => { cancelled = true; };
  }, [token, selectedId, requestWithToken]);

  useEffect(() => {
    if (!retrainJobId || !token || retrainJob?.state === "complete" || retrainJob?.state === "error") return undefined;
    let cancelled = false;
    let timer;
    const poll = async () => {
      try {
        const latest = await requestWithToken(`/api/jobs/${retrainJobId}`);
        if (cancelled) return;
        setRetrainJob(latest);
        if (latest.state === "running") timer = window.setTimeout(poll, 1200);
        else setTrainingBusy(false);
      } catch (requestError) {
        if (!cancelled) { setError(requestError.message); setTrainingBusy(false); }
      }
    };
    poll();
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [retrainJobId, retrainJob?.state, token, requestWithToken]);

  async function signIn(event) {
    event.preventDefault();
    setLoginBusy(true);
    setError("");
    try {
      const data = await apiRequest("/api/doctor/login", jsonOptions({ pin }));
      window.sessionStorage.setItem("prakriti-doctor-token", data.token);
      setToken(data.token);
      setPin("");
    } catch (requestError) { setError(requestError.message); }
    finally { setLoginBusy(false); }
  }

  async function signOut() {
    if (token) {
      try { await requestWithToken("/api/doctor/logout", { method: "POST", headers: { "X-Doctor-Token": token } }); } catch { /* Local sign-out still clears the session. */ }
    }
    window.sessionStorage.removeItem("prakriti-doctor-token");
    window.sessionStorage.removeItem("prakriti-retrain-job");
    setToken("");
    setSummary(null);
    setSubject(null);
    setForm(null);
    setRetrainJobId("");
    setRetrainJob(null);
    setError("");
  }

  async function saveVerification(event) {
    event.preventDefault();
    if (!subject || !form) return;
    setSaving(true);
    setError("");
    try {
      await requestWithToken("/api/doctor/verify", jsonOptions({
        subject_id: subject.id,
        dosha: form.dosha,
        features: form.features,
        notes: form.notes,
      }));
      await refreshSummary();
      setSuccessMessage(`Saved ${subject.id} as ${form.dosha}.`);
      setError("");
    } catch (requestError) { setError(requestError.message); }
    finally { setSaving(false); }
  }

  async function startRetraining() {
    setTrainingBusy(true);
    setRetrainJob(null);
    setError("");
    try {
      const data = await requestWithToken("/api/doctor/retrain", { method: "POST", headers: { "X-Doctor-Token": token } });
      setRetrainJobId(data.job_id);
      window.sessionStorage.setItem("prakriti-retrain-job", data.job_id);
      setRetrainJob({ state: "running", progress: 0, message: "Starting model training…" });
    } catch (requestError) { setError(requestError.message); setTrainingBusy(false); }
  }

  const activeQueue = summary ? (queueMode === "pending" ? summary.pending : summary.verified) : [];

  return (
    <>
      <PageIntro eyebrow="Practitioner workspace" title="Clinical review" description="Review participant records, verify clinical observations, and manage model training." />

      {!token ? (
        <section className="login-layout">
          <div className="login-art"><span className="login-art-ring ring-a" /><span className="login-art-ring ring-b" /><span className="login-art-center"><Icon name="shield" size={34} /></span><div className="login-art-copy"><strong>Clinical workspace</strong><small>Practitioner access</small></div></div>
          <form className="content-card login-card" onSubmit={signIn}>
            <span className="login-lock"><Icon name="lock" size={22} /></span>
            <h2>Practitioner sign in</h2>
            <p>Enter your access code to review and verify participant records.</p>
            <label className="field"><span>Access code</span><input type="password" value={pin} onChange={(event) => setPin(event.target.value)} placeholder="Enter your PIN" autoComplete="current-password" /></label>
            {error && <div className="inline-error" role="alert">{error}</div>}
            <button className="button button-primary button-large full-width" type="submit" disabled={loginBusy}>{loginBusy ? <><span className="spinner" />Signing in…</> : <>Enter portal <Icon name="arrow" size={18} /></>}</button>
            <small className="login-hint">This workspace is intended for authorized practitioners.</small>
          </form>
        </section>
      ) : (
        <>
          <div className="doctor-toolbar"><div><span className="eyebrow">Participant validation</span><h2>Clinical workspace</h2></div><button className="button button-secondary signout-button" onClick={signOut}><Icon name="logout" size={16} /> Sign out</button></div>
          <div className="doctor-metrics">
            <MetricCard label="Verified cohort" value={summary?.verified.length ?? "—"} icon="check" />
            <MetricCard label="Pending review" value={summary?.pending.length ?? "—"} icon="clock" />
            <MetricCard label="Duplicates excluded" value={summary?.duplicates.length ?? "—"} icon="file" />
            <MetricCard label="Active model" value={summary ? "RBF-SVM" : "—"} icon="chart" />
          </div>

          <div className="portal-tabs" role="tablist" aria-label="Practitioner tools">
            <button role="tab" aria-selected={tab === "review"} className={tab === "review" ? "active" : ""} onClick={() => setTab("review")}><Icon name="clipboard" size={17} /> Review participants</button>
            <button role="tab" aria-selected={tab === "retrain"} className={tab === "retrain" ? "active" : ""} onClick={() => setTab("retrain")}><Icon name="refresh" size={17} /> Retrain model</button>
          </div>

          {tab === "review" ? (
            <section className="review-workspace">
                <div className="review-heading"><div><h2>Participant review queue</h2><p>Compare the photo record with the model estimate, then enter the practitioner’s verified findings.</p></div>
                <div className="queue-toggle"><button className={queueMode === "pending" ? "active" : ""} onClick={() => { setQueueMode("pending"); setSelectedId(""); setSuccessMessage(""); }}>Pending <span>{summary?.pending.length ?? 0}</span></button><button className={queueMode === "verified" ? "active" : ""} onClick={() => { setQueueMode("verified"); setSelectedId(""); setSuccessMessage(""); }}>Verified <span>{summary?.verified.length ?? 0}</span></button></div>
              </div>
              {successMessage && <div className="success-banner" role="status"><Icon name="check" size={17} />{successMessage}</div>}
              {summary && activeQueue.length === 0 ? <div className="empty-queue"><span><Icon name="check" size={25} /></span><h3>{queueMode === "pending" ? "All caught up" : "No verified records yet"}</h3><p>{queueMode === "pending" ? "There are no participants waiting for review." : "Verified participants will appear here for inspection."}</p></div> : (
                <div className="review-selector"><label className="field"><span>Select participant</span><select value={selectedId} onChange={(event) => { setSelectedId(event.target.value); setSuccessMessage(""); }}><option value="" disabled>Choose a participant</option>{activeQueue.map((entry) => <option key={entry.id} value={entry.id}>{entry.id} — {entry.name || "Unknown"} {entry.status ? `(${entry.status})` : ""}</option>)}</select></label><button className="icon-button refresh-button" aria-label="Refresh review queue" onClick={refreshSummary}><Icon name="refresh" size={17} /></button></div>
              )}

              {subjectLoading && <div className="loading-panel"><span className="spinner spinner-dark" /> Loading participant photos and details…</div>}
              {subject && form && !subjectLoading && <>
                <div className="participant-meta"><span className="participant-id"><Icon name="user" size={16} /> {subject.id}</span><span>{subject.name}</span><span>{subject.age} years</span><span>{subject.sex}</span></div>
                <div className="review-columns">
                  <section className="content-card participant-photos"><div className="panel-title"><span className="panel-icon"><Icon name="photo" size={18} /></span><div><h3>Participant photos</h3><p>Four uploaded hand views</p></div></div><div className="doctor-photo-grid">{VIEW_CONFIG.map((view) => <div className="doctor-photo" key={view.key}><span>{view.title}</span>{subject.images?.[view.key] ? <img src={subject.images[view.key]} alt={view.title} /> : <div className="missing-photo"><Icon name="photo" size={21} /><small>Image not available</small></div>}</div>)}</div></section>

                  <form className="content-card verification-form" onSubmit={saveVerification}>
                    <div className="panel-title"><span className="panel-icon"><Icon name="clipboard" size={18} /></span><div><h3>Clinical verification</h3><p>Review the model estimate and enter verified findings</p></div></div>
                    <div className="ai-estimate"><span className="ai-estimate-icon"><Icon name="sparkle" size={17} /></span><div><small>MODEL ESTIMATE</small><strong>{subject.ml_predicted_dosha || "Not available"}</strong><span>{subject.ml_confidence_str || "Confidence details not available"}</span></div></div>
                    <label className="field"><span>Practitioner-verified Prakriti</span><select value={form.dosha} onChange={(event) => setForm((previous) => ({ ...previous, dosha: event.target.value }))}>{["K-P", "P-K", "K-V", "V-K", "P-V", "V-P", "K", "P", "V"].map((option) => <option key={option} value={option}>{option}</option>)}</select><small className="field-help">Order matters: for example, K-P and P-K indicate different dominant doshas.</small></label>
                    <div className="dosha-meaning"><strong>Clinical meaning</strong><span>{doshaMeaning(form.dosha)}</span></div>
                    <div className="form-divider" />
                    <div className="form-subheading"><h4>Seven clinical features</h4><small>Pre-filled from participant details and computer vision</small></div>
                    <div className="verification-grid">
                      <ObservationField label="Vein visibility" value={form.features.vein_visibility} name="vein_visibility" options={["Low", "Med", "High"]} onChange={(name, selection) => setForm((previous) => ({ ...previous, features: { ...previous.features, [name]: selection } }))} />
                      <ObservationField label="Mole on skin" value={form.features.mole} name="mole" options={["No", "Yes"]} onChange={(name, selection) => setForm((previous) => ({ ...previous, features: { ...previous.features, [name]: selection } }))} />
                      <ObservationField label="Skin texture" value={form.features.skin_texture} name="skin_texture" options={["Smooth", "Rough", "Dry", "Oily"]} onChange={(name, selection) => setForm((previous) => ({ ...previous, features: { ...previous.features, [name]: selection } }))} />
                      <ObservationField label="Skin color" value={form.features.skin_color} name="skin_color" options={["Fair", "Medium", "Dark"]} onChange={(name, selection) => setForm((previous) => ({ ...previous, features: { ...previous.features, [name]: selection } }))} />
                      <ObservationField label="Palm color" value={form.features.palm_color} name="palm_color" options={["PinkWhite", "PinkRed", "Brown"]} onChange={(name, selection) => setForm((previous) => ({ ...previous, features: { ...previous.features, [name]: selection } }))} />
                      <ObservationField label="Hair thickness" value={form.features.hair_thickness} name="hair_thickness" options={["Thin", "Med", "Thick"]} onChange={(name, selection) => setForm((previous) => ({ ...previous, features: { ...previous.features, [name]: selection } }))} />
                      <ObservationField label="Hair color" value={form.features.hair_color} name="hair_color" options={["Black", "Brown"]} onChange={(name, selection) => setForm((previous) => ({ ...previous, features: { ...previous.features, [name]: selection } }))} />
                    </div>
                    <label className="field notes-field"><span>Clinical notes</span><textarea rows="3" value={form.notes} onChange={(event) => setForm((previous) => ({ ...previous, notes: event.target.value }))} placeholder="Add a note for this verification" /></label>
                    {error && <div className="inline-error" role="alert"><Icon name="info" size={17} />{error}</div>}
                    <button className="button button-primary button-large full-width" type="submit" disabled={saving}>{saving ? <><span className="spinner" />Saving verification…</> : <>Confirm and save verification <Icon name="check" size={17} /></>}</button>
                    <small className="sync-note"><Icon name="cloud" size={15} /> Saves to the validation sheet, participant folder, and connected cloud record.</small>
                  </form>
                </div>
              </>}
              {error && !subject && <div className="inline-error" role="alert"><Icon name="info" size={17} />{error}</div>}
            </section>
          ) : (
            <section className="content-card retrain-panel">
              <div className="retrain-copy"><span className="retrain-icon"><Icon name="chart" size={23} /></span><span className="eyebrow">Verified cohort only</span><h2>Retrain the model</h2><p>Use all clinician-verified participants to train the current 70-feature pipeline. The workflow evaluates the model with stratified 5-fold cross-validation before updating the active model.</p>
                <div className="cohort-summary"><span><Icon name="check" size={16} /> Available verified records</span><strong>{summary?.verified.length ?? 0}</strong></div>
                <button className="button button-primary button-large" onClick={startRetraining} disabled={trainingBusy || !summary?.verified.length}>{trainingBusy ? <><span className="spinner" />Training in progress…</> : <>Start model retraining <Icon name="arrow" size={18} /></>}</button>
                {error && <div className="inline-error" role="alert">{error}</div>}
              </div>
              <div className="retrain-visual"><div className="retrain-ring"><span><Icon name="refresh" size={31} /></span><i /><i /><i /></div><div className="retrain-visual-note"><strong>Clinical ground truth</strong><small>Practitioner-verified cohort</small></div></div>
              {retrainJob && <div className="retrain-status"><div className="job-progress-title">{retrainJob.state === "running" && <span className="spinner spinner-dark" />}<div><strong>{retrainJob.state === "running" ? "Retraining in progress" : retrainJob.result?.success ? "Model retrained" : "Training needs attention"}</strong><small>{retrainJob.message || retrainJob.error || ""}</small></div><b>{retrainJob.progress ?? 0}%</b></div><div className="progress-track"><span style={{ width: `${retrainJob.progress ?? 0}%` }} /></div>
                {retrainJob.state === "complete" && retrainJob.result?.success && <><div className="training-metrics"><MetricCard label="Balanced accuracy" value={`${(retrainJob.result.balanced_accuracy * 100).toFixed(1)}%`} icon="chart" /><MetricCard label="Raw accuracy" value={`${(retrainJob.result.accuracy * 100).toFixed(1)}%`} icon="chart" /><MetricCard label="Macro F1" value={`${(retrainJob.result.macro_f1 * 100).toFixed(1)}%`} icon="chart" /></div><div className="training-cohort"><strong>Training cohort · {retrainJob.result.n_samples} participants</strong>{Object.entries(retrainJob.result.class_distribution || {}).map(([label, count]) => <span key={label}>{DOSHA_INFO[label]?.name || label} <b>{count}</b></span>)}</div></>}
                {retrainJob.error && <div className="inline-error" role="alert">{retrainJob.error}</div>}
              </div>}
            </section>
          )}
        </>
      )}
    </>
  );
}

function doshaMeaning(value) {
  const [first, second] = value.split("-");
  const firstName = DOSHA_INFO[first]?.name || value;
  if (!second) return `${firstName} dominant.`;
  const secondName = DOSHA_INFO[second]?.name;
  return `${firstName} dominant, ${secondName} sub-dominant.`;
}

export default App;
