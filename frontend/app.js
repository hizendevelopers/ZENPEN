const root = document.getElementById('app');

const USERS_KEY = 'media-suite-users';
const SESSION_KEY = 'media-suite-session';

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
    return JSON.parse(localStorage.getItem(USERS_KEY) || '[]');
  } catch {
    return [];
  }
}

function saveUsers(users) {
  localStorage.setItem(USERS_KEY, JSON.stringify(users));
}

function getSession() {
  try {
    return JSON.parse(localStorage.getItem(SESSION_KEY) || 'null');
  } catch {
    return null;
  }
}

function setSession(session) {
  localStorage.setItem(SESSION_KEY, JSON.stringify(session));
}

function clearSession() {
  localStorage.removeItem(SESSION_KEY);
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
        <span class="user-chip">${escapeHtml(session.name || session.email)}</span>
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
        <span class="brand-kicker">Media Suite</span>
        <strong>Content Intelligence Platform</strong>
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
    <section class="landing-hero">
      <div class="hero-copy">
        <span class="eyebrow">Before Login Experience</span>
        <h1>Analyze media, detect topics, and generate publication-ready content in one workspace.</h1>
        <p>
          Start with URL or video analysis, move into topic selection, and finish with structured articles and subtitle-ready workflows.
        </p>
        <div class="hero-actions">
          <a class="primary-btn" href="#/signup">Get Started</a>
          <a class="ghost-btn" href="#/login">Login</a>
        </div>
      </div>
      <div class="hero-panel">
        <div class="mini-stat">
          <strong>3</strong>
          <span>Products available</span>
        </div>
        <div class="mini-stat">
          <strong>2-Step</strong>
          <span>Analyze then generate</span>
        </div>
        <div class="mini-stat">
          <strong>Responsive</strong>
          <span>Desktop and mobile ready</span>
        </div>
      </div>
    </section>

    <section class="section-block">
      <div class="section-heading">
        <span class="eyebrow">Products</span>
        <h2>Explore the product suite</h2>
        <p>Use the dropdown in the header or jump into a product after login.</p>
      </div>
      <div class="product-grid">
        ${PRODUCTS.map((product) => buildProductCard(product, !isAuthenticated())).join('')}
      </div>
    </section>

    <section class="info-grid">
      <article class="info-card">
        <h3>About Us</h3>
        <p>We build media intelligence tools that turn raw video and audio into usable publishing assets.</p>
      </article>
      <article class="info-card">
        <h3>Blogs</h3>
        <p>Read product thinking, workflow ideas, and editorial automation insights from the team.</p>
      </article>
      <article class="info-card">
        <h3>Contact Us</h3>
        <p>Need implementation help, integration support, or feature planning? Reach out from the contact page.</p>
      </article>
    </section>
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
    <section class="dashboard-hero">
      <div>
        <span class="eyebrow">After Login Dashboard</span>
        <h1>Welcome, ${escapeHtml(session?.name || session?.email || 'User')}</h1>
        <p>Choose a product below to continue your workflow.</p>
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
  return `
    <article class="product-card">
      <div class="product-banner ${escapeHtml(product.banner)}"></div>
      <div class="product-copy">
        <span class="product-status ${product.status === 'ready' ? 'status-ready' : 'status-soon'}">${product.status === 'ready' ? 'Available Now' : 'Coming Soon'}</span>
        <h3>${escapeHtml(product.title)}</h3>
        <p>${escapeHtml(product.description)}</p>
        <a class="primary-btn" href="${href}">${escapeHtml(product.cta)}</a>
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
    body: formData,
  });
  const payload = await response.json();
  if (!response.ok || !payload.success) {
    throw new Error(payload.error || payload.detail || 'Request failed.');
  }
  return payload.result;
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

function handleLoginSubmit(form) {
  const formData = new FormData(form);
  const email = String(formData.get('email') || '').trim().toLowerCase();
  const password = String(formData.get('password') || '');
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

function handleSignupSubmit(form) {
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
} else {
  renderApp();
}
