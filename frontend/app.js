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
  analysisResult: null,
  analysisError: '',
  articleError: '',
  busyMode: '',
  busyMessage: '',
  formValues: {
    url: '',
    query: 'Give breaking news and main points',
    articleCount: '1',
  },
  lastSubmission: null,
  selectedTopics: [],
};

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
  const hash = window.location.hash.replace(/^#/, '') || '/';
  return hash.startsWith('/') ? hash : `/${hash}`;
}

function navigate(path) {
  window.location.hash = path;
}

function isAuthenticated() {
  return Boolean(getSession());
}

async function initializeApp() {
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
  appState.busyMode = '';
  appState.busyMessage = '';
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
        <a class="ghost-btn" href="#/login">Login</a>
        <a class="primary-btn" href="#/signup">Sign Up</a>
      </div>
    `;

  return `
    <header class="site-header">
      <a class="brand-mark" href="#/${session ? 'dashboard' : ''}">
        <img class="brand-logo-image" src="/static/assets/zenpen-logo.png" alt="ZenPen logo" />
      </a>
      <nav class="top-nav">
        <details class="nav-dropdown">
          <summary>Products</summary>
          <div class="dropdown-menu">
            ${PRODUCTS.map((product) => `
              <a href="#/products/${product.slug}">
                <strong>${escapeHtml(product.title)}</strong>
                <span>${escapeHtml(product.description)}</span>
              </a>
            `).join('')}
          </div>
        </details>
        <a href="#/about">About Us</a>
        <a href="#/blogs">Blogs</a>
        <a href="#/contact">Contact Us</a>
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
          <a class="primary-btn" href="#/signup">Get Started</a>
          <a class="ghost-btn" href="#/products/article-generator">Explore Products</a>
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
          <a href="#/">Home</a>
          <a href="#/about">About Us</a>
          <a href="#/blogs">Blogs</a>
        </div>
        <div>
          <span class="footer-title">Products</span>
          <a href="#/products/article-generator">Article Generator</a>
          <a href="#/products/blog-generator">Blog Generator</a>
          <a href="#/products/srt-file-generator">.SRT File Generator</a>
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
  const switchLink = isLogin ? '#/signup' : '#/login';
  const switchLabel = isLogin ? 'Sign Up' : 'Login';
  const authError = appState.authError ? `<div class="state-banner error-state">${escapeHtml(appState.authError)}</div>` : '';

  return `
    <section class="auth-shell">
      <div class="auth-card">
        <span class="eyebrow">${isLogin ? 'Login' : 'Sign Up'}</span>
        <h1>${title}</h1>
        <p>${subtitle}</p>
        ${authError}
        <form id="${isLogin ? 'login-form' : 'signup-form'}" class="auth-form">
          ${isLogin ? '' : `
            <label>
              Full Name
              <input name="name" type="text" placeholder="Enter your full name" required />
            </label>
          `}
          <label>
            Email
            <input name="email" type="email" placeholder="name@example.com" required />
          </label>
          <label>
            Password
            <input name="password" type="password" placeholder="Enter your password" required />
          </label>
          ${isLogin ? '' : `
            <label>
              Confirm Password
              <input name="confirmPassword" type="password" placeholder="Confirm your password" required />
            </label>
          `}
          <button class="primary-btn wide-btn" type="submit">${isLogin ? 'Login' : 'Create Account'}</button>
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
          <a class="primary-btn" href="#/products/article-generator">Open Article Generator</a>
          <a class="ghost-btn" href="#/products/blog-generator">Explore Tools</a>
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
  const href = requireLogin ? '#/login' : `#/products/${product.slug}`;
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
      <a class="ghost-btn" href="#/dashboard">Back to Dashboard</a>
    </section>
  `;
}

function buildAnalysisSummary() {
  if (appState.busyMode === 'analyzing') {
    return `<div class="state-panel loading-state"><div class="loader"></div><p>${escapeHtml(appState.busyMessage)}</p></div>`;
  }
  if (appState.analysisError) {
    return `<div class="state-panel error-state">${escapeHtml(appState.analysisError)}</div>`;
  }
  if (!appState.analysisResult) {
    return `<div class="state-panel empty-state">No analysis yet. Submit a URL or upload a video to begin.</div>`;
  }

  const result = appState.analysisResult;
  const summaryItems = parseSummaryToList(result.summary);
  return `
    <div class="analysis-result-card">
      <span class="eyebrow">Analysis Output</span>
      <h2>${escapeHtml(result.headline)}</h2>
      <div class="summary-block">
        <h3>Summary</h3>
        <ul>
          ${summaryItems.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}
        </ul>
      </div>
    </div>
  `;
}

function buildTopicSelector() {
  if (!appState.analysisResult?.topics?.length) {
    return '';
  }
  const topics = appState.analysisResult.topics;
  const selectedSet = new Set(appState.selectedTopics.length ? appState.selectedTopics : topics.slice(0, 3));
  const articleState = appState.busyMode === 'articles'
    ? `<div class="state-banner loading-state inline-state"><div class="loader small-loader"></div><span>${escapeHtml(appState.busyMessage)}</span></div>`
    : appState.articleError
      ? `<div class="state-banner error-state">${escapeHtml(appState.articleError)}</div>`
      : '';

  return `
    <section class="topic-section">
      <div class="section-heading compact-heading">
        <span class="eyebrow">Detected Topics</span>
        <h3>Select topics for article generation</h3>
      </div>
      <form id="topic-selection-form" class="topic-form">
        <div class="topic-selector-list">
          ${topics.map((topic) => `
            <label class="topic-option">
              <input type="checkbox" name="selected-topic" value="${escapeHtml(topic)}" ${selectedSet.has(topic) ? 'checked' : ''} />
              <span class="topic-pill">${escapeHtml(topic)}</span>
            </label>
          `).join('')}
        </div>
        <div class="topic-actions">
          <label>
            Articles per topic
            <input id="article-count-input" name="article_count" type="number" min="1" max="3" value="${escapeHtml(appState.formValues.articleCount)}" />
          </label>
          <button class="primary-btn" type="submit" ${appState.busyMode === 'articles' ? 'disabled' : ''}>Generate Articles</button>
        </div>
        ${articleState}
      </form>
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
        <button class="ghost-btn" data-action="download-all-articles">Download All</button>
      </div>
      <div class="articles-grid">
        ${articles.map((article, index) => `
          <article class="article-output-card">
            <div class="article-image-wrap">
              <img src="${escapeHtml(article.image_url || '')}" alt="${escapeHtml(article.topic)} article image" />
            </div>
            <div class="article-output-body">
              <div class="article-card-header">
                <div>
                  <span class="topic-tag">${escapeHtml(article.topic)}</span>
                  <h4>${escapeHtml(appState.analysisResult.headline)}</h4>
                </div>
                <div class="article-actions">
                  <button class="secondary-btn" data-action="copy-article" data-article-index="${index}">Copy</button>
                  <button class="secondary-btn" data-action="download-article" data-article-index="${index}">Download</button>
                </div>
              </div>
              <div class="article-content">${escapeHtml(article.content).replace(/\n/g, '<br />')}</div>
            </div>
          </article>
        `).join('')}
      </div>
    </section>
  `;
}

function buildArticleGeneratorPage() {
  return `
    <section class="generator-hero">
      <div>
        <span class="eyebrow">Product Page</span>
        <h1>Article Generator</h1>
        <p>Enter a URL or upload a video file, analyze the content, then generate topic-specific articles.</p>
      </div>
      <a class="ghost-btn" href="#/dashboard">Back to Dashboard</a>
    </section>

    <section class="generator-grid">
      <div class="panel analysis-form-panel">
        <h2>Step 1: Analyze Source Content</h2>
        <form id="article-generator-form" class="product-form">
          <label>
            Enter a URL
            <input id="url-input" name="url" type="url" placeholder="https://www.youtube.com/watch?v=..." value="${escapeHtml(appState.formValues.url)}" />
          </label>
          <div class="or-divider">or</div>
          <label>
            Upload a video file
            <input id="file-input" name="file" type="file" accept="video/mp4,video/mov,video/avi,video/webm,video/mkv" />
          </label>
          <label>
            Analysis prompt
            <input id="query-input" name="query" type="text" value="${escapeHtml(appState.formValues.query)}" />
          </label>
          <button class="primary-btn wide-btn" type="submit" ${appState.busyMode === 'analyzing' ? 'disabled' : ''}>Analyze Content</button>
        </form>
      </div>

      <div class="analysis-output-column">
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
  const protectedRoutes = route === '/dashboard' || route.startsWith('/products/');
  if (protectedRoutes && !isAuthenticated()) {
    navigate('/login');
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
  if (!url && !file) return 'Please provide a URL or upload a video file.';
  if (url) {
    try {
      new URL(url);
    } catch {
      return 'Please enter a valid URL.';
    }
  }
  if (file && !file.type.startsWith('video/')) {
    return 'Unsupported video format. Please upload a valid video file.';
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

async function sendAnalyzeRequest({ generateArticle, selectedTopics, source }) {
  const formData = new FormData();
  if (source.url) formData.append('url', source.url);
  if (source.query) formData.append('query', source.query);
  if (source.file) formData.append('file', source.file);
  formData.append('generate_article', generateArticle ? 'true' : 'false');
  formData.append('article_count', appState.formValues.articleCount || '1');
  if (selectedTopics?.length) formData.append('selected_topics', selectedTopics.join(','));

  const response = await fetch('/api/analyze', {
    method: 'POST',
    headers: buildAuthHeaders(),
    body: formData,
  });
  const responseText = await response.text();
  let payload = null;
  try {
    payload = responseText ? JSON.parse(responseText) : null;
  } catch {
    payload = null;
  }
  if (!response.ok || !payload.success) {
    const message = payload?.error || payload?.detail || responseText || 'Request failed.';
    throw new Error(message);
  }
  return payload.result;
}

async function sendAuthRequest(path, payload) {
  const response = await fetch(path, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data.success) {
    throw new Error(data.detail || data.error || 'Authentication request failed.');
  }
  return data.session;
}

async function handleAnalyzeSubmit(form) {
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

  appState.formValues.url = url;
  appState.formValues.query = query;
  appState.analysisError = '';
  appState.articleError = '';
  appState.busyMode = 'analyzing';
  appState.busyMessage = 'Analyzing your URL or uploaded video...';
  renderApp();

  try {
    const source = { url, query, file };
    appState.lastSubmission = source;
    const result = await sendAnalyzeRequest({ generateArticle: false, selectedTopics: [], source });
    appState.analysisResult = result;
    appState.selectedTopics = result.topics ? result.topics.slice(0, 3) : [];
    appState.busyMode = '';
    appState.busyMessage = '';
    renderApp();
  } catch (error) {
    appState.analysisError = error.message;
    appState.busyMode = '';
    appState.busyMessage = '';
    renderApp();
  }
}

async function handleArticleGeneration(form) {
  if (!appState.lastSubmission) {
    appState.articleError = 'Analyze content first before generating articles.';
    renderApp();
    return;
  }

  const selectedTopics = Array.from(form.querySelectorAll('input[name="selected-topic"]:checked')).map((input) => input.value);
  if (!selectedTopics.length) {
    appState.articleError = 'Select at least one topic before generating articles.';
    renderApp();
    return;
  }

  appState.selectedTopics = selectedTopics;
  appState.formValues.articleCount = form.querySelector('#article-count-input').value || '1';
  appState.articleError = '';
  appState.busyMode = 'articles';
  appState.busyMessage = 'Generating full articles for your selected topics...';
  renderApp();

  try {
    const result = await sendAnalyzeRequest({
      generateArticle: true,
      selectedTopics,
      source: appState.lastSubmission,
    });
    appState.analysisResult = result;
    appState.busyMode = '';
    appState.busyMessage = '';
    renderApp();
  } catch (error) {
    appState.articleError = error.message;
    appState.busyMode = '';
    appState.busyMessage = '';
    renderApp();
  }
}

async function handleLoginSubmit(form) {
  const formData = new FormData(form);
  const email = String(formData.get('email') || '').trim().toLowerCase();
  const password = String(formData.get('password') || '');

  if (hasSupabaseAuth()) {
    try {
      const session = await sendAuthRequest('/api/auth/login', { email, password });
      setSession(session);
      appState.authError = '';
      navigate('/dashboard');
      return;
    } catch (error) {
      appState.authError = error.message || 'Invalid email or password.';
      renderApp();
      return;
    }
  }

  const users = getUsers();
  const user = users.find((item) => item.email === email && item.password === password);
  if (!user) {
    appState.authError = 'Invalid email or password.';
    renderApp();
    return;
  }
  setSession({ name: user.name, email: user.email });
  appState.authError = '';
  navigate('/dashboard');
}

async function handleSignupSubmit(form) {
  const formData = new FormData(form);
  const name = String(formData.get('name') || '').trim();
  const email = String(formData.get('email') || '').trim().toLowerCase();
  const password = String(formData.get('password') || '');
  const confirmPassword = String(formData.get('confirmPassword') || '');

  if (!name || !email || !password) {
    appState.authError = 'All fields are required.';
    renderApp();
    return;
  }
  if (password !== confirmPassword) {
    appState.authError = 'Passwords do not match.';
    renderApp();
    return;
  }

  if (hasSupabaseAuth()) {
    try {
      const session = await sendAuthRequest('/api/auth/signup', { name, email, password });
      setSession(session);
      appState.authError = '';
      navigate('/dashboard');
      return;
    } catch (error) {
      appState.authError = error.message || 'Could not create your account.';
      renderApp();
      return;
    }
  }

  const users = getUsers();
  if (users.some((user) => user.email === email)) {
    appState.authError = 'An account with this email already exists.';
    renderApp();
    return;
  }

  users.push({ name, email, password });
  saveUsers(users);
  setSession({ name, email });
  appState.authError = '';
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
  const text = `Article ${index + 1}: ${article.topic}\n\n${article.content}`;
  await navigator.clipboard.writeText(text);
}

function attachDelegatedHandlers() {
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
    if (event.target.id === 'topic-selection-form') {
      event.preventDefault();
      handleArticleGeneration(event.target);
    }
  });

  document.addEventListener('click', async (event) => {
    const button = event.target.closest('[data-action]');
    if (!button) return;

    if (button.dataset.action === 'logout') {
      clearSession();
      resetGeneratorState();
      navigate('/');
      return;
    }

    if (button.dataset.action === 'download-all-articles') {
      const articles = appState.analysisResult?.articles || [];
      const payload = articles
        .map((article, index) => `Article ${index + 1}: ${article.topic}\n\n${article.content}`)
        .join('\n\n' + '-'.repeat(80) + '\n\n');
      downloadTextFile('generated-articles.txt', payload);
      return;
    }

    if (button.dataset.action === 'copy-article') {
      const articleIndex = Number(button.dataset.articleIndex);
      const article = appState.analysisResult?.articles?.[articleIndex];
      if (!article) return;
      await copyArticle(article, articleIndex);
      return;
    }

    if (button.dataset.action === 'download-article') {
      const articleIndex = Number(button.dataset.articleIndex);
      const article = appState.analysisResult?.articles?.[articleIndex];
      if (!article) return;
      const safeTopic = article.topic.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || `article-${articleIndex + 1}`;
      downloadTextFile(`${safeTopic}.txt`, `Article ${articleIndex + 1}: ${article.topic}\n\n${article.content}`);
    }
  });
}

window.addEventListener('hashchange', () => {
  const route = currentRoute();
  if (!route.startsWith('/products/article-generator')) {
    resetGeneratorState();
  }
  renderApp();
});

attachDelegatedHandlers();
if (!window.location.hash) {
  navigate('/');
}
renderApp();
initializeApp();
