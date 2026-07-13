const root = document.getElementById('app');

const LOCAL_USERS_KEY = 'media-suite-users';
const LOCAL_SESSION_KEY = 'media-suite-session';

const PRODUCTS = [
  {
    slug: 'article-generator',
    title: 'Article Generator',
    description: 'Turn a video or URL into a headline, summary, topic list, and polished article output.',
    cta: 'Open Product',
    status: 'ready',
    banner: 'banner-article',
  },
  {
    slug: 'blog-generator',
    title: 'Blog Generator',
    description: 'Generate blog-ready long-form content, outlines, and editorial direction from source media.',
    cta: 'Coming Soon',
    status: 'coming-soon',
    banner: 'banner-blog',
  },
  {
    slug: 'srt-file-generator',
    title: '.SRT File Generator',
    description: 'Produce subtitle-ready caption files and timeline exports for publishing workflows.',
    cta: 'Coming Soon',
    status: 'coming-soon',
    banner: 'banner-srt',
  },
];

const appState = {
  initialized: false,
  config: null,
  supabase: null,
  session: null,
  authError: '',
  authMode: 'login',
  authBusy: '',
  authBusyMessage: '',
  authForms: {
    login: { email: '', password: '' },
    signup: { name: '', email: '', password: '', confirmPassword: '' },
  },
  analysisResult: null,
  analysisError: '',
  articleError: '',
  publishError: '',
  busyMode: '',
  busyMessage: '',
  exportBusy: '',
  exportMessage: '',
  successMessage: '',
  activeJobId: '',
  activeJobStage: '',
  sourceMode: 'url',
  formValues: {
    url: '',
    query: 'Give breaking news and main points',
    articleCount: '1',
    articleType: 'Blog Article',
    targetAudience: 'General readers',
  },
  lastSubmission: null,
  selectedTopics: [],
};

let analysisPollTimer = null;
let busyStatusTimer = null;

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function getUsers() {
  try {
    return JSON.parse(localStorage.getItem(LOCAL_USERS_KEY) || '[]');
  } catch {
    return [];
  }
}

function saveUsers(users) {
  localStorage.setItem(LOCAL_USERS_KEY, JSON.stringify(users));
}

function getLocalSession() {
  try {
    return JSON.parse(localStorage.getItem(LOCAL_SESSION_KEY) || 'null');
  } catch {
    return null;
  }
}

function setLocalSession(session) {
  localStorage.setItem(LOCAL_SESSION_KEY, JSON.stringify(session));
}

function clearLocalSession() {
  localStorage.removeItem(LOCAL_SESSION_KEY);
}

function hasSupabaseAuth() {
  return Boolean(appState.config?.supabase?.enabled);
}

function getSession() {
  return appState.session || getLocalSession();
}

function setSession(session) {
  appState.session = session;
  setLocalSession(session);
}

function clearSession() {
  appState.session = null;
  clearLocalSession();
}

function getUserDisplayName(session = getSession()) {
  if (!session) return '';
  if (hasSupabaseAuth()) {
    return session.user?.user_metadata?.full_name || session.user?.email || 'User';
  }
  return session.name || session.email || 'User';
}

function currentRoute() {
  return window.location.pathname || '/';
}

function navigate(path, { replace = false } = {}) {
  const target = path.startsWith('/') ? path : `/${path}`;
  if (window.location.pathname === target) {
    renderApp();
    return;
  }
  if (replace) {
    window.history.replaceState({}, '', target);
  } else {
    window.history.pushState({}, '', target);
  }
  renderApp();
}

function isAuthenticated() {
  return Boolean(getSession());
}

async function initializeApp() {
  if (window.location.hash.startsWith('#/')) {
    const migrated = window.location.hash.replace(/^#/, '');
    window.history.replaceState({}, '', migrated);
  }

  try {
    const response = await fetch('/api/config');
    if (response.ok) {
      appState.config = await response.json();
    }
  } catch {
    appState.config = null;
  }

  appState.session = getLocalSession();

  appState.initialized = true;
  renderApp();
}

function productBySlug(slug) {
  return PRODUCTS.find((product) => product.slug === slug);
}

function resetGeneratorState() {
  appState.analysisResult = null;
  appState.analysisError = '';
  appState.articleError = '';
  appState.publishError = '';
  appState.successMessage = '';
  clearBusyStates();
  appState.selectedTopics = [];
  appState.lastSubmission = null;
}

function parseSummaryToList(summary) {
  return String(summary || '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.replace(/^-+\s*/, ''));
}

function stripHtmlToText(html) {
  const temp = document.createElement('div');
  temp.innerHTML = String(html || '');
  return temp.textContent || temp.innerText || '';
}

function setPageMetadata(route) {
  const metaByRoute = {
    '/': {
      title: 'ZENPEN | AI Article, Blog & Subtitle Workflows',
      description: 'Turn videos, URLs, and uploads into polished articles, blogs, and subtitle-ready outputs with AI-assisted workflows.',
    },
    '/login': {
      title: 'Login | ZENPEN',
      description: 'Login to access your AI publishing workspace and continue your content workflow.',
    },
    '/signup': {
      title: 'Sign Up | ZENPEN',
      description: 'Create your ZENPEN account to analyze media and generate structured content.',
    },
    '/dashboard': {
      title: 'Dashboard | ZENPEN',
      description: 'Open your tools, manage outputs, and continue your content generation workflow.',
    },
    '/products/article-generator': {
      title: 'Article Generator | ZENPEN',
      description: 'Generate professional articles from YouTube videos, media uploads, and URLs with AI-assisted analysis.',
    },
  };
  const fallback = {
    title: 'ZENPEN',
    description: 'AI-powered media analysis and publishing workflows.',
  };
  const meta = metaByRoute[route] || fallback;
  document.title = meta.title;
  let description = document.querySelector('meta[name="description"]');
  if (!description) {
    description = document.createElement('meta');
    description.name = 'description';
    document.head.appendChild(description);
  }
  description.setAttribute('content', meta.description);
}

function mapUserFacingError(error) {
  const message = String(error?.message || error || '').trim();
  const lower = message.toLowerCase();
  if (!message) return 'Something went wrong. Please try again.';
  if (lower.includes('passwords do not match')) return 'Passwords do not match. Please check them and try again.';
  if (lower.includes('invalid email or password')) return 'We could not log you in with those details.';
  if (lower.includes('please provide')) return 'Please provide a supported source before continuing.';
  if (lower.includes('unsupported')) return 'That source type is not supported for this action.';
  if (lower.includes('video could not be downloaded')) return 'Video could not be downloaded. The source may be private or unavailable.';
  if (lower.includes('source is private or unavailable')) return 'Source is private or unavailable.';
  if (lower.includes('gemini could not analyze the video directly')) return 'Direct video analysis failed. Switching to Gemini transcription...';
  if (lower.includes('direct video analysis took too long')) return 'Direct video analysis took too long. Switching to Gemini transcription...';
  if (lower.includes('transcription failed')) return 'Transcription failed. Please try another source or upload the audio/video file directly.';
  if (lower.includes('captions were not available')) return 'Captions were not available. Trying audio transcription...';
  if (lower.includes('upload the audio/video file or try another link') || lower.includes('could not extract reliable content from this video')) return 'We could not extract the video automatically. Please upload the audio/video file or try another link.';
  if (lower.includes('youtube blocked automated extraction')) return 'YouTube blocked automated extraction for this video. Please upload the audio/video file.';
  if (lower.includes('website blocked automated access')) return 'The website blocked automated access. Please try another public URL.';
  if (lower.includes('could not provide transcript or subtitle text quickly enough')) return 'Direct video analysis failed. Trying audio transcription fallback...';
  if (lower.includes('youtube video cannot be accessed')) return 'This YouTube video cannot be processed right now. Please try another source or upload the file.';
  if (lower.includes('timed out')) return 'Processing timed out. Please retry or try a different source.';
  if (lower.includes('rate') && lower.includes('busy')) return 'The AI service is busy right now. Please retry in a moment.';
  if (/\b\d{3}\b/.test(message) || lower.includes('traceback') || lower.includes('runtimeerror')) {
    return 'We could not complete that request right now. Please try again.';
  }
  return message;
}

function stopAnalysisPolling() {
  if (analysisPollTimer) {
    clearTimeout(analysisPollTimer);
    analysisPollTimer = null;
  }
  appState.activeJobId = '';
  appState.activeJobStage = '';
}

function stopBusyStatusTimer() {
  if (busyStatusTimer) {
    clearTimeout(busyStatusTimer);
    busyStatusTimer = null;
  }
}

function startBusyStatusTimer(mode) {
  stopBusyStatusTimer();
  const stagesByMode = {
    analyzing: [
      { delay: 8000, message: 'Downloading video audio for Gemini transcription...' },
      { delay: 22000, message: 'Transcribing source with Gemini...' },
    ],
    articles: [
      { delay: 8000, message: 'Polishing article...' },
      { delay: 26000, message: 'This is taking longer than usual. We are optimizing the article from your source.' },
    ],
  };
  const stages = stagesByMode[mode] || [];
  let index = 0;
  const scheduleNext = () => {
    const stage = stages[index];
    if (!stage) return;
    busyStatusTimer = setTimeout(() => {
      if (appState.busyMode === mode) {
        appState.busyMessage = stage.message;
        renderApp();
      }
      index += 1;
      scheduleNext();
    }, stage.delay);
  };
  scheduleNext();
}

function syncSourceInputs() {
  const form = document.getElementById('article-generator-form');
  if (!form) return;
  const urlInput = form.querySelector('#url-input');
  const fileInput = form.querySelector('#file-input');
  const fileHelper = form.querySelector('#file-helper');
  const urlHelper = form.querySelector('#url-helper');
  const clearUrlButton = form.querySelector('[data-action="clear-url"]');
  const clearFileButton = form.querySelector('[data-action="clear-file"]');
  if (!(urlInput instanceof HTMLInputElement) || !(fileInput instanceof HTMLInputElement)) return;

  const hasUrl = Boolean(urlInput.value.trim());
  const hasFile = Boolean(fileInput.files?.length);

  urlInput.disabled = hasFile;
  fileInput.disabled = hasUrl;

  if (fileHelper) {
    fileHelper.textContent = hasUrl
      ? 'Clear the URL if you want to switch to file upload.'
      : 'Select an audio or video file if you do not want to use a URL.';
  }
  if (urlHelper) {
    urlHelper.textContent = hasFile
      ? 'Reset the selected file if you want to switch back to a URL.'
      : 'Paste a webpage or YouTube/video URL.';
  }
  if (clearUrlButton instanceof HTMLButtonElement) {
    clearUrlButton.disabled = !hasUrl;
  }
  if (clearFileButton instanceof HTMLButtonElement) {
    clearFileButton.disabled = !hasFile;
  }
}

function clearBusyStates() {
  appState.busyMode = '';
  appState.busyMessage = '';
  appState.authBusy = '';
  appState.authBusyMessage = '';
  appState.exportBusy = '';
  appState.exportMessage = '';
  stopBusyStatusTimer();
  stopAnalysisPolling();
}

function buildHeader() {
  const session = getSession();
  const authActions = session
    ? `
      <div class="header-auth">
        <span class="user-chip">${escapeHtml(getUserDisplayName(session))}</span>
        <button class="ghost-btn" data-action="logout">Logout</button>
      </div>
    `
    : `
      <div class="header-auth">
        <a class="ghost-btn" href="/login">Login</a>
        <a class="primary-btn" href="/signup">Sign Up</a>
      </div>
    `;

  return `
    <header class="site-header">
      <a class="brand-mark" href="${session ? '/dashboard' : '/'}">
        <img class="brand-logo-image" src="/static/assets/zenpen-logo.png" alt="ZenPen logo" />
      </a>
      <nav class="top-nav">
        <details class="nav-dropdown">
          <summary>Products</summary>
          <div class="dropdown-menu">
            ${PRODUCTS.map((product) => `
              <a href="/products/${product.slug}">
                <strong>${escapeHtml(product.title)}</strong>
                <span>${escapeHtml(product.description)}</span>
              </a>
            `).join('')}
          </div>
        </details>
        <a href="/about">About Us</a>
        <a href="/blogs">Blogs</a>
        <a href="/contact">Contact Us</a>
      </nav>
      ${authActions}
    </header>
  `;
}

function buildLandingPage() {
  return `
    <section class="landing-hero premium-hero hero-surface">
      <img class="hero-background-image" src="/static/assets/hero-ai-dashboard.png" alt="AI dashboard and robot assistant interface" />
      <div class="hero-surface-overlay"></div>
      <div class="hero-copy hero-overlay-card">
        <span class="eyebrow">AI-Powered Content Workflow</span>
        <h1>
          Create Articles,
          <br />
          Blogs & Subtitles
          <br />
          from Any Video
          <br />
          or URL
        </h1>
        <p>
          ZENPEN helps your team analyze media, uncover key topics, and create
          publication-ready content with a clean, guided AI workflow built for
          speed, clarity, and creative momentum.
        </p>
        <div class="hero-badges">
          <span>Fast content analysis</span>
          <span>Premium AI workflows</span>
          <span>Built for modern publishing teams</span>
        </div>
        <div class="hero-actions">
          <a class="primary-btn" href="/signup">Get Started</a>
          <a class="ghost-btn" href="/products/article-generator">Explore Products</a>
        </div>
        <div class="hero-trust-strip">
          <div>
            <strong>3 Products</strong>
            <span>ready for your content pipeline</span>
          </div>
          <div>
            <strong>AI-First</strong>
            <span>designed for content operations</span>
          </div>
        </div>
      </div>
      <div class="floating-tool-card hero-floating-note">
        <span class="eyebrow">Workflow Stack</span>
        <strong>Analyze. Select. Generate.</strong>
        <p>Move from raw media to structured output in a single polished workspace.</p>
      </div>
    </section>

    <section class="section-block homepage-section">
      <div class="section-heading">
        <span class="eyebrow">Our Products</span>
        <h2>Purpose-built AI tools for modern creators</h2>
        <p>Explore focused products designed to turn source media into usable editorial assets without friction.</p>
      </div>
      <div class="product-grid">
        ${PRODUCTS.map((product) => buildProductCard(product, !isAuthenticated())).join('')}
      </div>
    </section>

    <section class="feature-rail">
      <article class="feature-rail-card">
        <span class="feature-rail-icon icon-time"></span>
        <div>
          <h3>Save Time</h3>
          <p>Automate repetitive content work and move faster.</p>
        </div>
      </article>
      <article class="feature-rail-card">
        <span class="feature-rail-icon icon-ai"></span>
        <div>
          <h3>Enhance Creativity</h3>
          <p>Get AI suggestions that unlock more content ideas.</p>
        </div>
      </article>
      <article class="feature-rail-card">
        <span class="feature-rail-icon icon-growth"></span>
        <div>
          <h3>Improve Productivity</h3>
          <p>Keep your team aligned in one guided workflow.</p>
        </div>
      </article>
      <article class="feature-rail-card">
        <span class="feature-rail-icon icon-cap"></span>
        <div>
          <h3>Easy to Learn</h3>
          <p>Use a clean interface built for quick onboarding.</p>
        </div>
      </article>
    </section>

    <section class="section-block homepage-section feature-section">
      <div class="section-heading">
        <span class="eyebrow">Feature Highlights</span>
        <h2>Built to simplify modern content operations</h2>
        <p>From first analysis to final output, every step is designed to save time and keep your workflow easy to manage.</p>
      </div>
      <div class="feature-grid">
        <article class="feature-card">
          <span class="feature-icon icon-time"></span>
          <h3>Save Time</h3>
          <p>Reduce repetitive editorial work with guided AI automation across analysis and generation.</p>
        </article>
        <article class="feature-card">
          <span class="feature-icon icon-ai"></span>
          <h3>AI-Powered Content Generation</h3>
          <p>Generate headlines, summaries, topics, and long-form outputs from a single source workflow.</p>
        </article>
        <article class="feature-card">
          <span class="feature-icon icon-growth"></span>
          <h3>Improve Productivity</h3>
          <p>Give teams a centralized toolset that accelerates content decisions and publishing readiness.</p>
        </article>
        <article class="feature-card">
          <span class="feature-icon icon-star"></span>
          <h3>Easy to Use</h3>
          <p>Keep the experience approachable with a clean interface, clear actions, and responsive layouts.</p>
        </article>
      </div>
    </section>

    <section class="info-grid homepage-info-grid">
      <article class="info-card">
        <span class="eyebrow">About Us</span>
        <h3>AI tools with a publishing mindset</h3>
        <p>We build media intelligence products that help teams move faster from raw video and audio to polished deliverables.</p>
      </article>
      <article class="info-card">
        <span class="eyebrow">Blogs</span>
        <h3>Insights for modern editorial workflows</h3>
        <p>Discover product thinking, workflow improvements, and AI content strategies tailored for growing teams.</p>
      </article>
      <article class="info-card">
        <span class="eyebrow">Contact Us</span>
        <h3>Need a custom workflow?</h3>
        <p>Reach out for implementation help, tailored feature planning, or deeper integration support for your team.</p>
      </article>
    </section>
  `;
}

function buildFooter() {
  return `
    <footer class="site-footer">
      <div class="footer-brand">
        <img class="brand-logo-image footer-brand-logo" src="/static/assets/zenpen-logo.png" alt="ZenPen logo" />
        <div>
          <p>Professional AI-powered content workflows for modern publishing teams.</p>
        </div>
      </div>
      <div class="footer-links">
        <div>
          <span class="footer-title">Quick Links</span>
          <a href="/">Home</a>
          <a href="/about">About Us</a>
          <a href="/blogs">Blogs</a>
        </div>
        <div>
          <span class="footer-title">Products</span>
          <a href="/products/article-generator">Article Generator</a>
          <a href="/products/blog-generator">Blog Generator</a>
          <a href="/products/srt-file-generator">.SRT File Generator</a>
        </div>
        <div>
          <span class="footer-title">Contact</span>
          <a href="mailto:support@zenpen.ai">support@zenpen.ai</a>
          <span>Global AI workflow support</span>
          <span>Copyright (c) 2026 ZENPEN AI</span>
        </div>
      </div>
    </footer>
  `;
}

function buildAuthPage(mode) {
  const isLogin = mode === 'login';
  const title = isLogin ? 'Welcome back' : 'Create your account';
  const subtitle = isLogin
    ? 'Login to access your products and continue generating content.'
    : 'Sign up to unlock the dashboard and start using the product suite.';
  const switchText = isLogin ? 'Need an account?' : 'Already have an account?';
  const switchLink = isLogin ? '/signup' : '/login';
  const switchLabel = isLogin ? 'Sign Up' : 'Login';
  const busy = appState.authBusy === mode;
  const formState = isLogin ? appState.authForms.login : appState.authForms.signup;
  const authError = appState.authError ? `<div class="state-banner error-state">${escapeHtml(appState.authError)}</div>` : '';
  const authBusyBanner = busy ? `<div class="state-banner loading-state"><div class="loader small-loader"></div><span>${escapeHtml(appState.authBusyMessage || 'Working...')}</span></div>` : '';

  return `
    <section class="auth-shell">
      <div class="auth-card">
        <span class="eyebrow">${isLogin ? 'Login' : 'Sign Up'}</span>
        <h1>${title}</h1>
        <p>${subtitle}</p>
        ${authBusyBanner}
        ${authError}
        <form id="${isLogin ? 'login-form' : 'signup-form'}" class="auth-form">
          ${isLogin ? '' : `
            <label>
              Full Name
              <input name="name" type="text" placeholder="Enter your full name" value="${escapeHtml(formState.name || '')}" required />
            </label>
          `}
          <label>
            Email
            <input name="email" type="email" placeholder="name@example.com" value="${escapeHtml(formState.email || '')}" required />
          </label>
          <label>
            Password
            <input name="password" type="password" placeholder="Enter your password" value="${escapeHtml(formState.password || '')}" required />
          </label>
          ${isLogin ? '' : `
            <label>
              Confirm Password
              <input name="confirmPassword" type="password" placeholder="Confirm your password" value="${escapeHtml(formState.confirmPassword || '')}" required />
            </label>
          `}
          <button class="primary-btn wide-btn" type="submit" ${busy ? 'disabled' : ''}>${busy ? 'Please wait...' : (isLogin ? 'Login' : 'Create Account')}</button>
        </form>
        <p class="auth-switch">${switchText} <a href="${switchLink}">${switchLabel}</a></p>
      </div>
    </section>
  `;
}

function buildDashboard() {
  const session = getSession();
  return `
    <section class="dashboard-hero hero-surface after-login-hero">
      <img class="hero-background-image" src="/static/assets/hero-ai-dashboard.png" alt="AI dashboard and robot assistant interface" />
      <div class="hero-surface-overlay"></div>
      <div class="hero-copy hero-overlay-card dashboard-overlay-card">
        <span class="eyebrow">Your AI Workspace</span>
        <h1>
          Welcome back,
          <br />
          ${escapeHtml(getUserDisplayName(session))}
        </h1>
        <p>
          Pick up where you left off, launch the right tool in seconds, and move from source media to polished output inside one focused workspace.
        </p>
        <div class="dashboard-badges">
          <span>Ready to generate</span>
          <span>Fast workflow access</span>
          <span>Built for content teams</span>
        </div>
        <div class="hero-actions dashboard-actions">
          <a class="primary-btn" href="/products/article-generator">Open Article Generator</a>
          <a class="ghost-btn" href="/products/blog-generator">Explore Tools</a>
        </div>
        <div class="dashboard-mini-points">
          <div>
            <strong>3 Tools</strong>
            <span>available in your dashboard</span>
          </div>
          <div>
            <strong>AI-First</strong>
            <span>designed for fast content delivery</span>
          </div>
        </div>
      </div>
      <div class="floating-tool-card hero-floating-note dashboard-floating-note">
        <span class="eyebrow">Workspace Ready</span>
        <strong>Your tools are live</strong>
        <p>Jump into article generation, blog workflows, or subtitle preparation from your dashboard.</p>
      </div>
    </section>
    <section class="section-block">
      <div class="product-grid">
        ${PRODUCTS.map((product) => buildProductCard(product, false)).join('')}
      </div>
    </section>
  `;
}

function buildProductCard(product, requireLogin) {
  const href = requireLogin ? '/login' : `/products/${product.slug}`;
  const productMeta = {
    'article-generator': {
      badge: 'AG',
      badgeClass: 'badge-article',
      visualClass: 'visual-article',
      brand: 'Article AI',
      cta: 'Open Tool',
    },
    'blog-generator': {
      badge: 'BG',
      badgeClass: 'badge-blog',
      visualClass: 'visual-blog',
      brand: 'Blog Studio',
      cta: 'Try Soon',
    },
    'srt-file-generator': {
      badge: 'SR',
      badgeClass: 'badge-srt',
      visualClass: 'visual-srt',
      brand: 'Caption Flow',
      cta: 'Try Soon',
    },
  }[product.slug];
  return `
    <article class="product-card">
      <div class="product-copy">
        <div class="product-topline">
          <div class="product-brand">
            <span class="product-badge ${productMeta.badgeClass}">${productMeta.badge}</span>
            <div>
              <strong>${productMeta.brand}</strong>
              <span class="product-accent-line"></span>
            </div>
          </div>
          <span class="product-status ${product.status === 'ready' ? 'status-ready' : 'status-soon'}">${product.status === 'ready' ? 'Available Now' : 'Coming Soon'}</span>
        </div>
        <h3>${escapeHtml(product.title)}</h3>
        <p>${escapeHtml(product.description)}</p>
        <div class="product-showcase ${productMeta.visualClass}">
          <div class="showcase-window">
            <span class="window-dot"></span>
            <span class="window-dot"></span>
            <span class="window-dot"></span>
          </div>
          <div class="showcase-body">
            <span></span>
            <span></span>
            <span></span>
          </div>
          <div class="showcase-floating-card"></div>
        </div>
        <a class="primary-btn" href="${href}">${productMeta.cta}</a>
      </div>
    </article>
  `;
}

function buildInfoPage(title, body, kicker = 'Information') {
  return `
    <section class="simple-page">
      <span class="eyebrow">${escapeHtml(kicker)}</span>
      <h1>${escapeHtml(title)}</h1>
      <p>${escapeHtml(body)}</p>
    </section>
  `;
}

function buildComingSoonPage(product) {
  return `
    <section class="simple-page">
      <span class="eyebrow">Product Placeholder</span>
      <h1>${escapeHtml(product.title)}</h1>
      <p>${escapeHtml(product.description)}</p>
      <div class="state-banner empty-state">This product page is ready, but the full workflow is coming soon.</div>
      <a class="ghost-btn" href="/dashboard">Back to Dashboard</a>
    </section>
  `;
}

function buildAnalysisSummary() {
  if (appState.busyMode === 'analyzing') {
    return `<div class="state-panel loading-state"><div class="loader"></div><p>${escapeHtml(appState.busyMessage)}</p></div>`;
  }
  if (appState.analysisError) {
    return `
      <div class="state-panel error-state error-state-stack">
        <p>${escapeHtml(appState.analysisError)}</p>
        <div class="error-actions-row">
          <button class="ghost-btn" data-action="retry-analysis">Retry</button>
        </div>
      </div>
    `;
  }
  if (!appState.analysisResult) {
    return `<div class="state-panel empty-state">No analysis yet. Submit a URL or upload a video to begin.</div>`;
  }

  const result = appState.analysisResult;
  const summaryItems = parseSummaryToList(result.summary);
  const keyPoints = result.key_points?.length ? result.key_points : summaryItems;
  return `
    <div class="analysis-result-card">
      <span class="eyebrow">Analysis Output</span>
      <h2>${escapeHtml(result.heading || result.headline)}</h2>
      ${result.topic_generation_warning ? `<div class="state-banner empty-state inline-state">${escapeHtml(result.topic_generation_warning)}</div>` : ''}
      <div class="analysis-grid">
        <div class="summary-block">
          <h3>Summary</h3>
          <p>${escapeHtml(stripHtmlToText(result.summary || ''))}</p>
        </div>
        <div class="summary-block">
          <h3>Key Points</h3>
          <ul>
            ${keyPoints.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}
          </ul>
        </div>
      </div>
    </div>
  `;
}

function buildTopicSelector() {
  if (!appState.analysisResult?.topics?.length) {
    return '';
  }
  const topics = appState.analysisResult.topics;
  const topicDetails = appState.analysisResult.topic_details || topics.map((topic) => ({
    title: topic,
    explanation: '',
    importance: '',
  }));
  const selectedTopic = appState.selectedTopics[0] || topics[0];
  const articleState = appState.busyMode === 'articles'
    ? `<div class="state-banner loading-state inline-state"><div class="loader small-loader"></div><span>${escapeHtml(appState.busyMessage)}</span></div>`
    : appState.articleError
        ? `<div class="state-banner error-state">${escapeHtml(appState.articleError)}</div>`
      : '';

  return `
    <section class="topic-section">
      <div class="section-heading compact-heading">
        <span class="eyebrow">Detected Topics</span>
        <h3>Which topic do you want an article on?</h3>
      </div>
      <div class="topic-form">
        <div class="topic-selector-grid">
          ${topicDetails.map((topic) => `
            <article class="topic-option ${selectedTopic === topic.title ? 'topic-option-active' : ''}">
              <span class="topic-pill">${escapeHtml(topic.title)}</span>
              ${topic.summary || topic.explanation ? `<p class="topic-option-copy">${topic.summary || topic.explanation}</p>` : ''}
              ${Array.isArray(topic.points) && topic.points.length ? `
                <div class="topic-points-list">
                  ${topic.points.map((point) => `
                    <div class="topic-point">
                      <strong>${escapeHtml(point.label || 'Key idea')}:</strong>
                      <p>${escapeHtml(point.description || '')}</p>
                    </div>
                  `).join('')}
                </div>
              ` : ''}
              ${topic.importance ? `<p class="topic-option-note">${topic.importance}</p>` : ''}
              <button class="primary-btn topic-generate-btn" type="button" data-action="generate-topic-article" data-topic-title="${escapeHtml(topic.title)}" ${appState.busyMode === 'articles' ? 'disabled' : ''}>Generate Article</button>
            </article>
          `).join('')}
        </div>
        ${articleState}
      </div>
    </section>
  `;
}

function buildArticlesSection() {
  if (appState.busyMode === 'articles' && !appState.analysisResult?.articles?.length) {
    return `<div class="state-panel loading-state"><div class="loader"></div><p>${escapeHtml(appState.busyMessage)}</p></div>`;
  }
  const articles = appState.analysisResult?.articles || [];
  if (!articles.length) {
    return `<div class="state-panel empty-state">No articles generated yet. Select one or more topics and generate articles.</div>`;
  }

  return `
    <section class="articles-section">
      <div class="articles-header">
        <div>
          <span class="eyebrow">Generated Articles</span>
          <h3>Complete article output</h3>
        </div>
      </div>
      ${appState.exportMessage ? `<div class="state-banner loading-state inline-state"><div class="loader small-loader"></div><span>${escapeHtml(appState.exportMessage)}</span></div>` : ''}
      ${appState.successMessage ? `<div class="state-banner loading-state inline-state">${escapeHtml(appState.successMessage)}</div>` : ''}
      ${appState.publishError ? `<div class="state-banner error-state inline-state">${escapeHtml(appState.publishError)}</div>` : ''}
      <div class="articles-grid">
        ${articles.map((article, index) => `
          <article class="article-output-card">
            <div class="article-image-wrap">
              <img src="${escapeHtml(article.image_url || '')}" alt="${escapeHtml(article.topic)} article image" />
            </div>
            <div class="article-output-body">
              <div class="article-card-header article-card-header-inline">
                <div class="article-actions article-actions-left">
                  <button class="secondary-btn" data-action="copy-article" data-article-index="${index}" ${appState.exportBusy ? 'disabled' : ''}>Copy</button>
                  <details class="download-menu">
                    <summary class="secondary-btn" ${appState.exportBusy ? 'aria-disabled="true"' : ''}>Download</summary>
                    <div class="download-menu-list">
                      <button class="secondary-btn" type="button" data-action="download-docx" data-article-index="${index}" ${appState.exportBusy ? 'disabled' : ''}>DOCX</button>
                      <button class="secondary-btn" type="button" data-action="download-pdf" data-article-index="${index}" ${appState.exportBusy ? 'disabled' : ''}>PDF</button>
                    </div>
                  </details>
                </div>
                <div class="article-heading-block">
                  <span class="topic-tag">${escapeHtml(article.topic)}</span>
                  <h4>${escapeHtml(article.meta_title || appState.analysisResult.headline)}</h4>
                </div>
              </div>
              <div class="article-content article-content-wide">${article.content || ''}</div>
            </div>
            </article>
          `).join('')}
        </div>
    </section>
  `;
}

function buildArticleGeneratorPage() {
  const hasUrl = Boolean(appState.formValues.url?.trim());
  return `
    <section class="generator-hero">
      <div>
        <span class="eyebrow">Product Page</span>
        <h1>Article Generator</h1>
        <p>Enter a webpage or YouTube URL, or upload an audio/video file, then generate topic-specific articles in the format you need.</p>
      </div>
      <a class="ghost-btn" href="/dashboard">Back to Dashboard</a>
    </section>

    <section class="generator-stack">
      <div class="panel analysis-form-panel analysis-form-wide">
        <h2>Analyze Source Content</h2>
        <form id="article-generator-form" class="product-form">
          <div class="source-input-row">
            <label>
              Enter a URL
              <div class="input-with-action">
                <input id="url-input" name="url" type="url" placeholder="https://example.com/article or https://www.youtube.com/watch?v=..." value="${escapeHtml(appState.formValues.url)}" />
                <button class="ghost-btn inline-clear-btn" type="button" data-action="clear-url" ${hasUrl ? '' : 'disabled'}>Clear</button>
              </div>
              <span id="url-helper" class="field-helper">Paste a webpage or YouTube/video URL.</span>
            </label>
            <label>
              Upload an audio or video file
              <div class="input-with-action file-input-shell">
                <input id="file-input" name="file" type="file" accept="audio/*,video/*" />
                <button class="ghost-btn inline-clear-btn" type="button" data-action="clear-file">Reset</button>
              </div>
              <span id="file-helper" class="field-helper">Select an audio or video file if you do not want to use a URL.</span>
            </label>
          </div>
          <label>
            Analysis prompt
            <input id="query-input" name="query" type="text" value="${escapeHtml(appState.formValues.query)}" placeholder="Tell us what you want from this video or URL, e.g. 'Give me breaking news and main points'" />
          </label>
          <button class="primary-btn wide-btn" type="submit" ${appState.busyMode === 'analyzing' ? 'disabled' : ''}>Analyze Content</button>
          <p class="field-helper">Supported direct inputs: webpages, article URLs, YouTube/video links, and uploaded audio/video files. Image URLs are not supported yet.</p>
        </form>
      </div>

      <div class="analysis-output-column full-width-output">
        ${buildAnalysisSummary()}
        ${buildTopicSelector()}
        ${buildArticlesSection()}
      </div>
    </section>
  `;
}

function buildPageContent(route) {
  if (route === '/') return buildLandingPage();
  if (route === '/login') return buildAuthPage('login');
  if (route === '/signup') return buildAuthPage('signup');
  if (route === '/about') return buildInfoPage('About Us', 'We are building a content generation suite that helps teams analyze media and turn it into structured publishing outputs.', 'About Us');
  if (route === '/blogs') return buildInfoPage('Blogs', 'Our editorial blog space is ready for future product stories, launch notes, and workflow guides.', 'Blogs');
  if (route === '/contact') return buildInfoPage('Contact Us', 'Reach out for product help, custom workflows, or integration planning. This contact area is ready for a future form or CRM hookup.', 'Contact Us');
  if (route === '/dashboard') return buildDashboard();

  if (route.startsWith('/products/')) {
    const slug = route.split('/')[2];
    const product = productBySlug(slug);
    if (!product) return buildInfoPage('Not Found', 'The requested product page could not be found.', 'Error');
    if (product.slug === 'article-generator') return buildArticleGeneratorPage();
    return buildComingSoonPage(product);
  }

  return buildInfoPage('Not Found', 'The page you are looking for does not exist.', 'Error');
}

function renderApp() {
  if (!appState.initialized) {
    root.innerHTML = `
      <div class="page-shell">
        <main class="page-content">
          <div class="state-panel loading-state">
            <div class="loader"></div>
            <p>Loading your workspace...</p>
          </div>
        </main>
      </div>
    `;
    return;
  }

  const route = currentRoute();
  setPageMetadata(route);
  const protectedRoutes = route === '/dashboard' || route.startsWith('/products/');
  if (protectedRoutes && !isAuthenticated()) {
    navigate('/login', { replace: true });
    return;
  }

  root.innerHTML = `
    <div class="page-shell">
      ${buildHeader()}
      <main class="page-content">${buildPageContent(route)}</main>
      ${buildFooter()}
    </div>
  `;
}

function validateAnalysisInput(url, file) {
  if (!url && !file) return 'Please provide a URL or upload an audio/video file.';
  if (url) {
    try {
      new URL(url);
    } catch {
      return 'Please enter a valid URL.';
    }
  }
  if (file && file.type && !file.type.startsWith('video/') && !file.type.startsWith('audio/')) {
    return 'Unsupported file format. Please upload a valid audio or video file.';
  }
  return '';
}

function buildAuthHeaders() {
  if (hasSupabaseAuth() && appState.session?.access_token) {
    return {
      Authorization: `Bearer ${appState.session.access_token}`,
    };
  }
  return {};
}

async function sendAnalyzeRequest({ generateArticle, selectedTopics, source, endpoint = '/api/analyze' }) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 180000);
  const formData = new FormData();
  if (source.url) formData.append('url', source.url);
  if (source.query) formData.append('query', source.query);
  if (source.file) formData.append('file', source.file);
  formData.append('generate_article', generateArticle ? 'true' : 'false');
  formData.append('article_count', appState.formValues.articleCount || '1');
  formData.append('article_type', appState.formValues.articleType || 'Blog Article');
  formData.append('target_audience', appState.formValues.targetAudience || 'General readers');
  if (selectedTopics?.length) formData.append('selected_topics', selectedTopics.join(','));

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: endpoint === '/api/analyze-youtube'
        ? {
            'Content-Type': 'application/json',
            ...buildAuthHeaders(),
          }
        : buildAuthHeaders(),
      body: endpoint === '/api/analyze-youtube'
        ? JSON.stringify({
            url: source.url,
            query: source.query || 'Give breaking news and main points',
          })
        : formData,
      signal: controller.signal,
    });
    const responseText = await response.text();
    let payload = null;
    try {
      payload = responseText ? JSON.parse(responseText) : null;
    } catch {
      payload = null;
    }
    if (!response.ok || !payload?.success) {
      const message = payload?.error || payload?.detail || responseText || 'Request failed.';
      throw new Error(mapUserFacingError(message));
    }
    return payload;
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error('Analysis is taking longer than expected. Please retry or use a shorter source.');
    }
    throw new Error(mapUserFacingError(error));
  } finally {
    clearTimeout(timeout);
  }
}

async function sendGenerateArticlesRequest({ headline, summary, topics, selectedTopics, articleCount, selectedTopicDetails = null }) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch('/api/generate-article', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...buildAuthHeaders(),
      },
      body: JSON.stringify({
        headline,
        summary,
        topics,
        selected_topics: selectedTopics,
        selected_topic_details: selectedTopicDetails,
        article_count: Number(articleCount || 1),
        article_type: 'Blog Article',
        target_audience: 'General readers',
        source_context: appState.analysisResult?.source_context_preview || appState.analysisResult?.summary || '',
        source_cache_key: appState.analysisResult?.source_cache_key || '',
      }),
      signal: controller.signal,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.success) {
      throw new Error(mapUserFacingError(payload?.detail || payload?.error || 'Article generation failed.'));
    }
    return payload.result;
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error('This is taking longer than usual. We are optimizing the article from your source.');
    }
    throw new Error(mapUserFacingError(error));
  } finally {
    clearTimeout(timeout);
  }
}

async function fetchAnalysisJob(jobId) {
  const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {
    headers: buildAuthHeaders(),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok && !payload?.queued) {
    throw new Error(mapUserFacingError(payload?.detail || payload?.error || 'Background analysis failed.'));
  }
  return payload;
}

function startAnalysisPolling(jobId) {
  stopAnalysisPolling();
  appState.activeJobId = jobId;
  const genericProcessingMessage = 'We are processing your source in the background.';

  const poll = async () => {
    try {
      const payload = await fetchAnalysisJob(jobId);
      if (payload.progress?.message && payload.progress.message !== genericProcessingMessage) {
        appState.busyMessage = payload.progress.message;
        appState.activeJobStage = payload.progress.stage || '';
      }
      if (payload.status === 'completed' && payload.result) {
        appState.analysisResult = payload.result;
        appState.selectedTopics = payload.result.topics ? payload.result.topics.slice(0, 1) : [];
        clearBusyStates();
        renderApp();
        return;
      }
      if (payload.status === 'failed' || payload.success === false) {
        appState.analysisError = mapUserFacingError(payload.message || payload.error || 'The background analysis failed.');
        clearBusyStates();
        renderApp();
        return;
      }
      analysisPollTimer = setTimeout(poll, 2500);
      renderApp();
    } catch (error) {
      appState.analysisError = mapUserFacingError(error);
      clearBusyStates();
      renderApp();
    }
  };

  analysisPollTimer = setTimeout(poll, 1200);
}

async function sendAuthRequest(path, payload) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 25000);
  try {
    const response = await fetch(path, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.success) {
      throw new Error(mapUserFacingError(data.detail || data.error || 'Authentication request failed.'));
    }
    return data.session;
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error('The request timed out. Please try again.');
    }
    throw new Error(mapUserFacingError(error));
  } finally {
    clearTimeout(timeout);
  }
}

async function exportArticle(article, format) {
  const response = await fetch('/api/articles/export', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...buildAuthHeaders(),
    },
    body: JSON.stringify({
      title: article.meta_title || appState.analysisResult?.headline || article.topic,
      topic: article.topic,
      content_html: article.content || '',
      format,
    }),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(mapUserFacingError(payload?.detail || payload?.error || 'Export failed.'));
  }
  const blob = await response.blob();
  const downloadUrl = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = downloadUrl;
  link.download = `${(article.slug || article.topic || 'generated-article').replace(/[^a-z0-9-]+/gi, '-')}.${format}`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(downloadUrl);
}

async function publishArticleDraft(article) {
  const response = await fetch('/api/articles/publish', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...buildAuthHeaders(),
    },
    body: JSON.stringify({
      headline: appState.analysisResult?.headline || article.meta_title || article.topic,
      summary: appState.analysisResult?.summary || '',
      topics: appState.analysisResult?.topics || [],
      selected_topics: [article.topic],
      articles: [article],
      source_type: appState.lastSubmission?.file ? 'upload' : 'url',
      source_url: appState.lastSubmission?.url || null,
      source_file_name: appState.lastSubmission?.file?.name || null,
      source_mime_type: appState.lastSubmission?.file?.type || null,
      query: appState.formValues.query || 'Give breaking news and main points',
    }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || !payload.success) {
    throw new Error(mapUserFacingError(payload?.detail || payload?.error || 'Draft publishing failed.'));
  }
  return payload;
}

async function handleAnalyzeSubmit(form) {
  if (appState.busyMode === 'analyzing') return;
  const fileInput = form.querySelector('#file-input');
  const url = form.querySelector('#url-input').value.trim();
  const query = form.querySelector('#query-input').value.trim() || 'Give breaking news and main points';
  const file = fileInput?.files?.[0] || null;
  const validationMessage = validateAnalysisInput(url, file);
  if (validationMessage) {
    appState.analysisError = validationMessage;
    appState.articleError = '';
    renderApp();
    return;
  }

  const sourceMode = file ? 'upload' : 'url';
  appState.sourceMode = sourceMode;
  appState.formValues.url = url;
  appState.formValues.query = query;
  appState.analysisError = '';
  appState.articleError = '';
  appState.publishError = '';
  appState.successMessage = '';
  appState.busyMode = 'analyzing';
  const youtubeMode = sourceMode !== 'upload' && url && /youtu(\.be|be\.com)/i.test(url);
  appState.busyMessage = sourceMode === 'upload'
    ? 'Uploading your media for Gemini transcription...'
    : youtubeMode
      ? 'Downloading video audio for Gemini transcription...'
      : 'Preparing source and extracting content...';
  startBusyStatusTimer('analyzing');
  renderApp();

    try {
      const source = { url, query, file };
      appState.lastSubmission = source;
      const payload = await sendAnalyzeRequest({
        generateArticle: false,
        selectedTopics: [],
        source,
        endpoint: youtubeMode ? '/api/analyze-youtube' : '/api/analyze',
      });
      if (payload.queued && payload.jobId) {
        appState.busyMode = 'analyzing';
        appState.busyMessage = payload.progress?.message || 'Downloading video audio for Gemini transcription...';
        appState.activeJobStage = payload.progress?.stage || '';
        startAnalysisPolling(payload.jobId);
        renderApp();
        return;
      }
      const result = payload.result;
      appState.analysisResult = result;
      appState.selectedTopics = result.topics ? result.topics.slice(0, 1) : [];
      clearBusyStates();
      renderApp();
    } catch (error) {
      appState.analysisError = mapUserFacingError(error);
      clearBusyStates();
      renderApp();
  }
}

async function handleDirectTopicArticleGeneration(topic) {
  if (appState.busyMode === 'articles') return;
  if (!appState.analysisResult) {
    appState.articleError = 'Analyze content first before generating an article.';
    renderApp();
    return;
  }
  appState.selectedTopics = [topic];
  appState.formValues.articleCount = '1';
  appState.articleError = '';
  appState.publishError = '';
  appState.successMessage = '';
  appState.busyMode = 'articles';
  appState.busyMessage = 'Writing article for selected topic...';
  startBusyStatusTimer('articles');
  renderApp();

  try {
    const selectedTopicDetails = (appState.analysisResult?.topic_details || []).find((item) => item.title === topic) || null;
    const result = await sendGenerateArticlesRequest({
      headline: appState.analysisResult?.headline || 'Media summary',
      summary: appState.analysisResult?.summary || '',
      topics: appState.analysisResult?.topics || [],
      selectedTopics: [topic],
      articleCount: 1,
      selectedTopicDetails,
    });
    appState.analysisResult = result;
    clearBusyStates();
    renderApp();
  } catch (error) {
    appState.articleError = mapUserFacingError(error);
    clearBusyStates();
    renderApp();
  }
}

async function handleLoginSubmit(form) {
  if (appState.authBusy === 'login') return;
  const formData = new FormData(form);
  const email = String(formData.get('email') || '').trim().toLowerCase();
  const password = String(formData.get('password') || '');
  appState.authForms.login = { email, password };
  appState.authBusy = 'login';
  appState.authBusyMessage = 'Signing you in...';
  appState.authError = '';
  renderApp();

  if (hasSupabaseAuth()) {
    try {
      const session = await sendAuthRequest('/api/auth/login', { email, password });
      setSession(session);
      appState.authError = '';
      clearBusyStates();
      appState.authForms.login = { email: '', password: '' };
      navigate('/dashboard');
      return;
    } catch (error) {
      appState.authError = mapUserFacingError(error) || 'Invalid email or password.';
      clearBusyStates();
      renderApp();
      return;
    }
  }

  const users = getUsers();
  const user = users.find((item) => item.email === email && item.password === password);
  if (!user) {
    appState.authError = 'Invalid email or password.';
    clearBusyStates();
    renderApp();
    return;
  }
  setSession({ name: user.name, email: user.email });
  appState.authError = '';
  clearBusyStates();
  appState.authForms.login = { email: '', password: '' };
  navigate('/dashboard');
}

async function handleSignupSubmit(form) {
  if (appState.authBusy === 'signup') return;
  const formData = new FormData(form);
  const name = String(formData.get('name') || '').trim();
  const email = String(formData.get('email') || '').trim().toLowerCase();
  const password = String(formData.get('password') || '');
  const confirmPassword = String(formData.get('confirmPassword') || '');
  appState.authForms.signup = { name, email, password, confirmPassword };

  if (!name || !email || !password) {
    appState.authError = 'All fields are required.';
    renderApp();
    return;
  }
  if (password !== confirmPassword) {
    appState.authError = 'Passwords do not match. Please recheck both password fields.';
    renderApp();
    return;
  }

  appState.authBusy = 'signup';
  appState.authBusyMessage = 'Creating your account...';
  appState.authError = '';
  renderApp();

  if (hasSupabaseAuth()) {
    try {
      const session = await sendAuthRequest('/api/auth/signup', { name, email, password });
      setSession(session);
      appState.authError = '';
      clearBusyStates();
      appState.authForms.signup = { name: '', email: '', password: '', confirmPassword: '' };
      navigate('/dashboard');
      return;
    } catch (error) {
      appState.authError = mapUserFacingError(error) || 'Could not create your account.';
      clearBusyStates();
      renderApp();
      return;
    }
  }

  const users = getUsers();
  if (users.some((user) => user.email === email)) {
    appState.authError = 'An account with this email already exists.';
    clearBusyStates();
    renderApp();
    return;
  }

  users.push({ name, email, password });
  saveUsers(users);
  setSession({ name, email });
  appState.authError = '';
  clearBusyStates();
  appState.authForms.signup = { name: '', email: '', password: '', confirmPassword: '' };
  navigate('/dashboard');
}

function downloadTextFile(filename, content) {
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function copyArticle(article, index) {
  const text = `Article ${index + 1}: ${article.topic}\n\n${stripHtmlToText(article.content)}`;
  await navigator.clipboard.writeText(text);
}

function attachDelegatedHandlers() {
  document.addEventListener('change', (event) => {
    const target = event.target;
    if (target instanceof HTMLInputElement && target.id === 'file-input') {
      syncSourceInputs();
    }
  });

  document.addEventListener('input', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement) && !(target instanceof HTMLTextAreaElement)) return;
    const form = target.form;
    if (!form) return;
    if (form.id === 'login-form' && target instanceof HTMLInputElement) {
      appState.authForms.login[target.name] = target.value;
    }
    if (form.id === 'signup-form' && target instanceof HTMLInputElement) {
      appState.authForms.signup[target.name] = target.value;
    }
    if (form.id === 'article-generator-form') {
      if (target.id === 'url-input') {
        appState.formValues.url = target.value;
        syncSourceInputs();
      }
      if (target.id === 'query-input') {
        appState.formValues.query = target.value;
      }
    }
  });

  document.addEventListener('submit', (event) => {
    if (event.target.id === 'login-form') {
      event.preventDefault();
      handleLoginSubmit(event.target);
    }
    if (event.target.id === 'signup-form') {
      event.preventDefault();
      handleSignupSubmit(event.target);
    }
    if (event.target.id === 'article-generator-form') {
      event.preventDefault();
      handleAnalyzeSubmit(event.target);
    }
  });

  document.addEventListener('click', async (event) => {
    const link = event.target.closest('a[href]');
    if (link) {
      const href = link.getAttribute('href') || '';
      if (href.startsWith('/') && !href.startsWith('//')) {
        event.preventDefault();
        clearBusyStates();
        navigate(href);
        return;
      }
    }

    const button = event.target.closest('[data-action]');
    if (!button) return;

    if (button.dataset.action === 'logout') {
      clearSession();
      resetGeneratorState();
      navigate('/');
      return;
    }

    if (button.dataset.action === 'retry-analysis') {
      if (appState.lastSubmission) {
        appState.analysisError = '';
        renderApp();
        const form = document.getElementById('article-generator-form');
        if (form) {
          handleAnalyzeSubmit(form);
        }
      }
      return;
    }

    if (button.dataset.action === 'generate-topic-article') {
      const topicTitle = button.dataset.topicTitle || '';
      if (topicTitle) {
        handleDirectTopicArticleGeneration(topicTitle);
      }
      return;
    }

    if (button.dataset.action === 'copy-article') {
      const articleIndex = Number(button.dataset.articleIndex);
      const article = appState.analysisResult?.articles?.[articleIndex];
      if (!article) return;
      await copyArticle(article, articleIndex);
      appState.successMessage = 'Article content copied to clipboard.';
      renderApp();
      return;
    }

      if (button.dataset.action === 'download-docx' || button.dataset.action === 'download-pdf') {
        const articleIndex = Number(button.dataset.articleIndex);
        const article = appState.analysisResult?.articles?.[articleIndex];
        if (!article) return;
        appState.exportBusy = button.dataset.action;
        appState.exportMessage = 'Preparing your export...';
        appState.publishError = '';
        renderApp();
        try {
          await exportArticle(article, button.dataset.action === 'download-docx' ? 'docx' : 'pdf');
          appState.successMessage = `${button.dataset.action === 'download-docx' ? 'DOCX' : 'PDF'} export is ready.`;
        } catch (error) {
          appState.publishError = mapUserFacingError(error);
        } finally {
          appState.exportBusy = '';
          appState.exportMessage = '';
          renderApp();
        }
        return;
      }

      if (button.dataset.action === 'clear-url') {
        const urlInput = document.getElementById('url-input');
        if (urlInput instanceof HTMLInputElement) {
          urlInput.value = '';
          appState.formValues.url = '';
          syncSourceInputs();
        }
        return;
      }

      if (button.dataset.action === 'clear-file') {
        const fileInput = document.getElementById('file-input');
        if (fileInput instanceof HTMLInputElement) {
          fileInput.value = '';
          syncSourceInputs();
        }
      }
  });
}

attachDelegatedHandlers();
window.addEventListener('popstate', () => {
  clearBusyStates();
  const route = currentRoute();
  if (!route.startsWith('/products/article-generator')) {
    resetGeneratorState();
  }
  renderApp();
});
if (!window.location.pathname) {
  navigate('/', { replace: true });
}
const __renderApp = renderApp;
renderApp = function wrappedRenderApp() {
  __renderApp();
  window.requestAnimationFrame(() => {
    syncSourceInputs();
  });
};
renderApp();
initializeApp();
