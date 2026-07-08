import io

from fastapi.testclient import TestClient

from app import (
    analyze_text_content,
    app,
    analyze_url_source,
    download_audio,
    extract_youtube_video_id,
    fetch_youtube_transcript_text,
    find_http_urls_in_payload,
    is_youtube_url,
    select_any_non_youtube_url,
    select_apify_download_url,
    select_best_audio_format_id,
)

client = TestClient(app)


def test_select_best_audio_format_prefers_audio_only():
    info = {
        'formats': [
            {'format_id': '18', 'acodec': 'aac', 'vcodec': 'h264', 'abr': 96, 'tbr': 300, 'ext': 'mp4'},
            {'format_id': '140', 'acodec': 'aac', 'vcodec': 'none', 'abr': 128, 'tbr': 128, 'ext': 'm4a'},
            {'format_id': '251', 'acodec': 'opus', 'vcodec': 'none', 'abr': 160, 'tbr': 160, 'ext': 'webm'},
        ]
    }
    assert select_best_audio_format_id(info) == '251'


def test_select_best_audio_format_falls_back_to_muxed():
    info = {
        'formats': [
            {'format_id': '18', 'acodec': 'aac', 'vcodec': 'h264', 'abr': 96, 'tbr': 300, 'ext': 'mp4'},
            {'format_id': '22', 'acodec': 'aac', 'vcodec': 'h264', 'abr': 192, 'tbr': 800, 'ext': 'mp4'},
        ]
    }
    assert select_best_audio_format_id(info) == '22'


def test_select_apify_download_url_prefers_download_fields():
    item = {
        'url': 'https://www.youtube.com/watch?v=abc123',
        'downloadUrl': 'https://api.apify.com/v2/key-value-stores/example/records/video.mp3',
    }
    assert select_apify_download_url(item) == 'https://api.apify.com/v2/key-value-stores/example/records/video.mp3'


def test_is_youtube_url_detects_supported_hosts():
    assert is_youtube_url('https://www.youtube.com/watch?v=abc123') is True
    assert is_youtube_url('https://youtu.be/abc123') is True
    assert is_youtube_url('https://example.com/file.mp3') is False


def test_select_any_non_youtube_url_finds_nested_media_link():
    payload = {
        'url': 'https://www.youtube.com/watch?v=abc123',
        'nested': {
            'download': {
                'href': 'https://cdn.example.com/audio/file.mp3',
            }
        },
    }
    assert select_any_non_youtube_url(payload) == 'https://cdn.example.com/audio/file.mp3'


def test_find_http_urls_in_payload_collects_nested_urls():
    payload = {
        'a': 'https://example.com/a.mp3',
        'b': [{'c': 'https://example.com/b.mp4'}],
    }
    urls = find_http_urls_in_payload(payload)
    assert 'https://example.com/a.mp3' in urls
    assert 'https://example.com/b.mp4' in urls


def test_extract_youtube_video_id_supports_multiple_formats():
    assert extract_youtube_video_id('https://www.youtube.com/watch?v=abc123') == 'abc123'
    assert extract_youtube_video_id('https://youtu.be/xyz789') == 'xyz789'
    assert extract_youtube_video_id('https://www.youtube.com/shorts/short123') == 'short123'


def test_analyze_text_content_uses_fallback_summary_when_empty():
    result = analyze_text_content('', 'Summarize')
    assert result['headline'] == 'Key takeaways for: Summarize'


def test_fetch_youtube_transcript_text_uses_fetch_api(monkeypatch):
    class FakeSnippet:
        def __init__(self, text):
            self.text = text

    class FakeApi:
        def fetch(self, video_id, languages=('en',), preserve_formatting=False):
            return [FakeSnippet('Hello'), FakeSnippet('world')]

    monkeypatch.setattr('app.get_youtube_transcript_api_class', lambda: FakeApi)
    assert fetch_youtube_transcript_text('https://www.youtube.com/watch?v=abc123') == 'Hello world'


def test_download_audio_raises_apify_error_directly_for_youtube(monkeypatch, tmp_path):
    monkeypatch.setattr('app.APIFY_TOKEN', 'token')
    monkeypatch.setattr('app.download_audio_via_apify', lambda url, output_dir: (_ for _ in ()).throw(RuntimeError('apify failed')))

    try:
        download_audio('https://www.youtube.com/watch?v=abc123', str(tmp_path))
    except RuntimeError as exc:
        assert 'Apify YouTube download failed' in str(exc)
    else:
        raise AssertionError('Expected RuntimeError')


def test_analyze_endpoint_uses_transcript_fallback_for_youtube(monkeypatch):
    monkeypatch.setattr('app.queue_is_available', lambda: False)
    monkeypatch.setattr('app.download_audio', lambda url, output_dir: (_ for _ in ()).throw(RuntimeError('download failed')))
    monkeypatch.setattr('app.fetch_youtube_transcript_text', lambda url: 'Transcript from YouTube captions.')
    monkeypatch.setattr('app.get_embeddings', lambda chunks: (_ for _ in ()).throw(RuntimeError('skip embeddings')))
    monkeypatch.setattr('app.get_topics_from_summary', lambda summary: ['Topic A'])

    response = client.post(
        '/api/analyze',
        data={'url': 'https://www.youtube.com/watch?v=abc123', 'query': 'Summarize'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['success'] is True
    assert 'headline' in payload['result']
    assert payload['result']['topics'] == ['Topic A']


def test_analyze_endpoint_prefers_youtube_transcript_before_download(monkeypatch):
    monkeypatch.setattr('app.queue_is_available', lambda: False)
    monkeypatch.setattr('app.fetch_youtube_transcript_text', lambda url: 'Transcript available immediately.')
    monkeypatch.setattr('app.download_audio', lambda url, output_dir: (_ for _ in ()).throw(AssertionError('download should not be called')))
    monkeypatch.setattr('app.get_embeddings', lambda chunks: (_ for _ in ()).throw(RuntimeError('skip embeddings')))
    monkeypatch.setattr('app.get_topics_from_summary', lambda summary: ['Topic A'])

    response = client.post(
        '/api/analyze',
        data={'url': 'https://www.youtube.com/watch?v=abc123', 'query': 'Summarize'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['success'] is True
    assert payload['result']['topics'] == ['Topic A']


def test_health_endpoint():
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'
    assert 'dependencies' in response.json()
    assert response.json()['dependencies']['python']


def test_config_endpoint_exposes_public_config_only():
    response = client.get('/api/config')
    assert response.status_code == 200
    payload = response.json()
    assert 'supabase' in payload
    assert 'publishableKey' in payload['supabase']
    assert 'SECRET' not in str(payload).upper()


def test_login_endpoint_returns_session(monkeypatch):
    monkeypatch.setattr('app.supabase_is_configured', lambda: True)
    monkeypatch.setattr(
        'app.sign_in_supabase_user',
        lambda email, password: {
            'access_token': 'token-123',
            'user': {'email': email, 'user_metadata': {'full_name': 'Test User'}},
        },
    )

    response = client.post('/api/auth/login', json={'email': 'test@example.com', 'password': 'secret123'})

    assert response.status_code == 200
    payload = response.json()
    assert payload['success'] is True
    assert payload['session']['access_token'] == 'token-123'


def test_signup_endpoint_creates_account_and_returns_session(monkeypatch):
    monkeypatch.setattr('app.supabase_is_configured', lambda: True)
    monkeypatch.setattr('app.create_supabase_user', lambda name, email, password: {'id': 'user-1', 'email': email})
    monkeypatch.setattr(
        'app.sign_in_supabase_user',
        lambda email, password: {
            'access_token': 'signup-token',
            'user': {'email': email, 'user_metadata': {'full_name': 'Test User'}},
        },
    )

    response = client.post(
        '/api/auth/signup',
        json={'name': 'Test User', 'email': 'test@example.com', 'password': 'secret123'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['success'] is True
    assert payload['session']['access_token'] == 'signup-token'


def test_history_endpoint():
    response = client.get('/api/history')
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_frontend_deep_links_fallback_to_index():
    response = client.get('/products/article-generator')
    assert response.status_code == 200
    assert 'text/html' in response.headers['content-type']


def test_analyze_endpoint_returns_summary_for_uploaded_audio(monkeypatch):
    monkeypatch.setattr('app.queue_is_available', lambda: False)
    monkeypatch.setattr('app.transcribe_audio', lambda _: 'This is a short transcript. It has two sentences.')
    monkeypatch.setattr('app.get_embeddings', lambda chunks: (_ for _ in ()).throw(RuntimeError('skip embeddings')))
    monkeypatch.setattr('app.get_topics_from_summary', lambda summary: ['Topic A', 'Topic B'])

    response = client.post(
        '/api/analyze',
        data={'query': 'Summarize'},
        files={'file': ('sample.wav', io.BytesIO(b'fake wav bytes'), 'audio/wav')},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['success'] is True
    assert 'headline' in payload['result']
    assert 'summary' in payload['result']
    assert payload['result']['topics'] == ['Topic A', 'Topic B']
    assert payload['result']['articles'] == []


def test_analyze_endpoint_falls_back_when_transcript_is_empty(monkeypatch):
    monkeypatch.setattr('app.queue_is_available', lambda: False)
    monkeypatch.setattr('app.transcribe_audio', lambda _: '')
    monkeypatch.setattr('app.get_topics_from_summary', lambda summary: ['Main topic'])

    response = client.post(
        '/api/analyze',
        data={'query': 'Summarize'},
        files={'file': ('sample.wav', io.BytesIO(b'fake wav bytes'), 'audio/wav')},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['success'] is True
    assert payload['result']['headline'] == 'Key takeaways for: Summarize'
    assert payload['result']['topics'] == ['Main topic']


def test_analyze_endpoint_generates_articles_when_requested(monkeypatch):
    monkeypatch.setattr('app.queue_is_available', lambda: False)
    monkeypatch.setattr('app.transcribe_audio', lambda _: 'This is a short transcript. It has two sentences.')
    monkeypatch.setattr('app.get_embeddings', lambda chunks: (_ for _ in ()).throw(RuntimeError('skip embeddings')))
    monkeypatch.setattr('app.get_topics_from_summary', lambda summary: ['Topic A'])
    monkeypatch.setattr('app.generate_news_article', lambda headline, summary, topic=None: f'Article for {topic}')

    response = client.post(
        '/api/analyze',
        data={'query': 'Summarize', 'generate_article': 'true', 'article_count': '1'},
        files={'file': ('sample.wav', io.BytesIO(b'fake wav bytes'), 'audio/wav')},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['success'] is True
    assert payload['result']['articles'] == [{
        'topic': 'Topic A',
        'content': 'Article for Topic A',
        'image_url': 'https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&w=1200&q=80',
    }]


def test_analyze_url_source_returns_topics_without_articles(monkeypatch):
    monkeypatch.setattr('app.fetch_youtube_transcript_text', lambda url: 'Transcript available immediately.')
    monkeypatch.setattr('app.get_embeddings', lambda chunks: (_ for _ in ()).throw(RuntimeError('skip embeddings')))
    monkeypatch.setattr('app.get_topics_from_summary', lambda summary: ['Topic A', 'Topic B'])

    result = analyze_url_source('https://www.youtube.com/watch?v=abc123', 'Summarize')

    assert result['topics'] == ['Topic A', 'Topic B']
    assert result['articles'] == []


def test_analyze_endpoint_queues_url_jobs_when_queue_is_available(monkeypatch):
    class FakeJob:
        id = 'job-123'

    monkeypatch.setattr('app.background_url_jobs_available', lambda: True)
    monkeypatch.setattr('app.enqueue_url_analysis', lambda **kwargs: FakeJob())

    response = client.post(
        '/api/analyze',
        data={'url': 'https://www.youtube.com/watch?v=abc123', 'query': 'Summarize'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['success'] is True
    assert payload['queued'] is True
    assert payload['job_id'] == 'job-123'


def test_job_status_endpoint_returns_completed_result(monkeypatch):
    class FakeJob:
        result = {'headline': 'Ready'}

        def get_status(self, refresh=True):
            return 'finished'

    monkeypatch.setattr('app.fetch_job', lambda job_id: FakeJob())

    response = client.get('/api/jobs/job-123')

    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'completed'
    assert payload['result']['headline'] == 'Ready'


def test_generate_articles_endpoint_returns_articles(monkeypatch):
    monkeypatch.setattr('app.generate_news_article', lambda headline, summary, topic=None: f'Article for {topic}')

    response = client.post(
        '/api/articles',
        json={
            'headline': 'Summary headline',
            'summary': '- point 1',
            'topics': ['Topic A'],
            'selected_topics': ['Topic A'],
            'article_count': 1,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['success'] is True
    assert payload['result']['articles'][0]['content'] == 'Article for Topic A'
