import io

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'
    assert 'dependencies' in response.json()
    assert response.json()['dependencies']['python']


def test_history_endpoint():
    response = client.get('/api/history')
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_analyze_endpoint_returns_summary_for_uploaded_audio(monkeypatch):
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
