import io

from fastapi.testclient import TestClient

from app import (
    ANALYSIS_JOB_MAX_SECONDS,
    ENABLE_REMOTE_MEDIA_DOWNLOAD_FALLBACK,
    build_youtube_source_text,
    YOUTUBE_DIRECT_ANALYSIS_TIMEOUT,
    analyze_text_content,
    app,
    analyze_youtube_source,
    analyze_url_source,
    build_chunk_safe_analysis_input,
    build_content_cache_key,
    build_dynamic_subheadings,
    download_audio,
    extract_youtube_video_id,
    extract_webpage_content,
    fallback_article,
    fetch_youtube_transcript_text,
    find_http_urls_in_payload,
    build_local_topic_details,
    gemini_rag,
    generate_article_html,
    get_topic_details_from_summary,
    map_public_error_message,
    remove_timestamps,
    save_cached_source_content,
    sanitize_article_html,
    serialize_job_status,
    is_youtube_url,
    select_any_non_youtube_url,
    select_apify_download_url,
    select_best_audio_format_id,
    transcribe_audio_with_fallback,
)

client = TestClient(app)


LONG_VIDEO_TRANSCRIPT = (
    "China's development model is presented as a story of institutions, coordination, and long-term planning. "
    "The speaker explains that rapid transformation is not the result of one individual but of a wider system that keeps executing over decades. "
    "The video compares this consistency with countries where policies change too often and long-term projects stall. "
    "It also connects universities, infrastructure, industry, and competition as parts of one larger national machine. "
    "The argument keeps returning to execution, discipline, and macro-level thinking as the real drivers of growth."
)


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
    result, _ = analyze_text_content('', 'Summarize')
    assert 'Summarize' in result['headline']


def test_build_local_topic_details_creates_editorial_titles():
    summary = (
        "- Saudi Arabia is accelerating large infrastructure projects at unusual speed.\n"
        "- The discussion links execution, planning, and national ambition.\n"
        "- Public messaging is tied to visible results and long-term economic change."
    )
    details = build_local_topic_details(summary)

    assert details
    assert len(details[0]["title"].split()) >= 2
    assert "source material" not in details[0]["importance"].lower()
    assert isinstance(details[0]["points"], list)
    assert details[0]["points"]


def test_remove_timestamps_strips_common_formats():
    text = "[00:19] First line (03:59) and 12:42 are removed."
    cleaned = remove_timestamps(text)
    assert "[" not in cleaned
    assert "03:59" not in cleaned
    assert "12:42" not in cleaned


def test_gemini_rag_returns_structured_analysis_without_timestamps(monkeypatch):
    monkeypatch.setattr(
        'app.gemini_generate_text',
        lambda prompt: """
        {
          "heading": "Inside a Coordinated Growth Model",
          "summary": "The video explains how policy, infrastructure, and long-term planning reinforce each other.",
          "key_points": ["[00:19] Long-term planning matters", "Infrastructure scales national capacity"],
          "topics": [
            {
              "id": "topic_1",
              "title": "Long-Term Vision and Consistency",
              "summary": "The source links durable planning with stronger execution.",
              "importance": "It shows why consistency shapes national outcomes.",
              "points": [
                {"label": "Decades-long planning", "description": "(03:59) The video highlights policy continuity over long horizons."},
                {"label": "Execution discipline", "description": "Institutions are shown as carrying plans through delivery."}
              ]
            }
          ]
        }
        """
    )
    result = gemini_rag("source context", "Give me breaking news and main points")
    assert result["heading"] == "Inside a Coordinated Growth Model"
    assert all(":" not in point[:6] for point in result["key_points"])
    assert result["topics"][0]["summary"] == "The source links durable planning with stronger execution."
    assert result["topics"][0]["points"][0]["description"].startswith("The video highlights")


def test_gemini_rag_retries_when_metadata_pollution_is_detected(monkeypatch):
    calls = {"count": 0}

    def fake_generate_text(prompt):
        calls["count"] += 1
        if calls["count"] == 1:
            return """
            {
              "heading": "Mooroo | https | Instagram",
              "summary": "instagram.com/example https://youtube.com/watch?v=test",
              "key_points": ["Instagram", "https"],
              "topics": [{"title": "Https", "points": [{"label": "Link", "description": "instagram.com/example"}]}]
            }
            """
        return """
        {
          "heading": "China's Growth Relies on Coordinated Systems",
          "summary": "The video explains how institutions, infrastructure, and long-term planning work together.",
          "key_points": ["Execution is framed as a system-wide strength.", "Long-term policy continuity supports industrial growth."],
          "topics": [
            {
              "title": "The Power of a Coordinated System",
              "points": [
                {"label": "Execution discipline", "description": "The speaker ties visible progress to disciplined institutions."}
              ]
            }
          ]
        }
        """

    monkeypatch.setattr('app.gemini_generate_text', fake_generate_text)
    result = gemini_rag(LONG_VIDEO_TRANSCRIPT, "Give breaking news and main points", source_url="https://youtu.be/example")
    assert calls["count"] == 2
    assert "https" not in result["heading"].lower()
    assert result["topics"][0]["title"] == "The Power of a Coordinated System"


def test_fallback_article_avoids_banned_meta_phrases():
    html = fallback_article(
        "How Ambition Is Reshaping a National Development Strategy",
        "- The speaker highlights rapid project delivery.\n"
        "- The summary connects visible construction with political intent.\n"
        "- The material emphasizes long-term economic positioning.\n"
        "- Public expectations are rising alongside the pace of change.",
        topic="National Development Strategy",
    )

    assert "Why This Topic Stands Out" not in html
    assert "the selected topic is" not in html.lower()
    assert "this article discusses" not in html.lower()


def test_build_dynamic_subheadings_are_topic_aware():
    headings = build_dynamic_subheadings(
        "Saudi Infrastructure Expansion",
        [
            "Saudi Arabia is accelerating large infrastructure projects.",
            "The discussion connects speed, execution, and national ambition.",
            "Public messaging is tied to visible results and economic change.",
        ],
        "Blog Article",
    )

    assert len(headings) == 4
    assert any("Saudi Infrastructure Expansion" in heading or "Saudi" in heading for heading in headings)


def test_sanitize_article_html_wraps_orphan_text_in_paragraphs():
    html = sanitize_article_html("<h2>Title</h2>\nPlain paragraph text without tags.")
    assert "<p>Plain paragraph text without tags.</p>" in html


def test_get_topic_details_rewrites_generic_gemini_titles(monkeypatch):
    monkeypatch.setattr(
        "app.gemini_generate_text",
        lambda prompt: '[{"title":"Speaker","explanation":"The explanation covers enduring loyalty and the promises made in the relationship.","importance":"It matters because the language of loyalty shapes the whole message."}]',
    )

    details = get_topic_details_from_summary("- The lyrics promise unwavering loyalty.\n- The message revolves around fidelity and trust.")

    assert details
    assert details[0]["title"] != "Speaker"
    assert len(details[0]["title"].split()) >= 2


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


def test_transcribe_audio_with_fallback_uses_cached_transcript(monkeypatch):
    monkeypatch.setattr('app.load_cached_transcript', lambda cache_key: 'cached transcript ready')
    monkeypatch.setattr('app.split_audio_for_transcription', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('should not split audio')))

    transcript = transcribe_audio_with_fallback('demo.wav', cache_key='cached-key')

    assert transcript == 'cached transcript ready'


def test_build_chunk_safe_analysis_input_reuses_summary_cache(monkeypatch, tmp_path):
    monkeypatch.setattr('app.SUMMARY_CACHE_DIR', tmp_path)
    monkeypatch.setattr('app.LONG_TRANSCRIPT_THRESHOLD', 10)
    calls = {'count': 0}

    def fake_generate_text(prompt):
        calls['count'] += 1
        return '- cached point'

    monkeypatch.setattr('app.gemini_generate_text', fake_generate_text)
    transcript = 'This is the first sentence. This is the second sentence. This is the third sentence. This is the fourth sentence.'

    first = build_chunk_safe_analysis_input(transcript, 'Summarize')
    second = build_chunk_safe_analysis_input(transcript, 'Summarize')

    assert first == second
    assert calls['count'] >= 1
    assert calls['count'] <= len(first.splitlines())


def test_extract_webpage_content_uses_cached_source(monkeypatch, tmp_path):
    monkeypatch.setattr('app.SOURCE_CACHE_DIR', tmp_path)
    cache_key = build_content_cache_key('webpage', 'https://example.com/story')
    cached_payload = {
        'title': 'Cached title',
        'meta_description': 'Cached description',
        'headings': ['Cached heading'],
        'content': 'Cached readable content from the article body that is definitely long enough for extraction.',
    }
    save_cached_source_content(cache_key, cached_payload)
    monkeypatch.setattr('app.httpx.get', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('network should not run')))

    payload = extract_webpage_content('https://example.com/story')

    assert payload == cached_payload


def test_generate_article_html_skips_proofread_in_fast_mode(monkeypatch):
    monkeypatch.setattr('app.ENABLE_DEEP_ARTICLE_REFINEMENT', False)
    monkeypatch.setattr('app.gemini_generate_text', lambda prompt: '<h2>Headline</h2><h3>Section</h3><p>Body paragraph one.</p><h3>More</h3><p>Body paragraph two.</p><h3>End</h3><p>Body paragraph three.</p>')
    monkeypatch.setattr('app.proofread_article_html', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('proofread should be skipped')))

    html = generate_article_html(
        headline_text='Headline',
        summary_text='- Point one\n- Point two\n- Point three',
        topic='Fast Path',
        article_type='Blog Article',
        source_context='This source context explains the topic with enough detail for testing.',
        target_audience='General readers',
    )

    assert '<h2>Headline</h2>' in html


def test_analyze_endpoint_uses_transcript_fallback_for_youtube(monkeypatch, tmp_path):
    monkeypatch.setattr('app.ANALYSIS_CACHE_DIR', tmp_path / 'analysis-cache')
    monkeypatch.setattr('app.background_url_jobs_available', lambda: False)
    monkeypatch.setattr('app.fetch_youtube_metadata', lambda url: {'title': 'Video headline', 'description': 'Instagram link https://instagram.com/test', 'channel': 'Channel', 'categories': ['News'], 'tags': ['Policy']})
    monkeypatch.setattr('app.direct_gemini_youtube_analysis', lambda url, query, metadata=None: (_ for _ in ()).throw(RuntimeError('unsupported direct analysis')))
    monkeypatch.setattr('app.fetch_youtube_transcript_text', lambda url: LONG_VIDEO_TRANSCRIPT)
    monkeypatch.setattr('app.gemini_rag', lambda context, query, source_url='': {
        'heading': 'Systems, Planning, and National Coordination',
        'summary': 'The video argues that coordinated institutions and long-term planning drive national transformation.',
        'key_points': ['Institutions matter more than personalities.', 'Consistency supports industrial growth.'],
        'topics': [{'title': 'Coordinated Systems', 'points': [{'label': 'Execution', 'description': 'The video links growth to disciplined execution.'}]}],
    })

    response = client.post(
        '/api/analyze',
        data={'url': 'https://www.youtube.com/watch?v=abc123', 'query': 'Summarize'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['success'] is True
    assert 'headline' in payload['result']
    assert payload['result']['topics'] == ['Coordinated Systems']


def test_analyze_endpoint_prefers_youtube_transcript_before_download(monkeypatch, tmp_path):
    monkeypatch.setattr('app.ANALYSIS_CACHE_DIR', tmp_path / 'analysis-cache')
    monkeypatch.setattr('app.background_url_jobs_available', lambda: False)
    download_calls = {'count': 0}
    monkeypatch.setattr('app.fetch_youtube_metadata', lambda url: {'title': 'Video headline', 'description': 'Video description', 'channel': 'Channel', 'categories': ['News'], 'tags': ['Policy']})
    monkeypatch.setattr('app.direct_gemini_youtube_analysis', lambda url, query, metadata=None: (_ for _ in ()).throw(RuntimeError('unsupported direct analysis')))
    monkeypatch.setattr('app.fetch_youtube_transcript_text', lambda url: LONG_VIDEO_TRANSCRIPT)
    monkeypatch.setattr('app.download_audio', lambda url, output_dir: download_calls.__setitem__('count', download_calls['count'] + 1) or 'fake-audio.wav')
    monkeypatch.setattr('app.gemini_rag', lambda context, query, source_url='': {
        'heading': 'Systems, Planning, and National Coordination',
        'summary': 'The video argues that coordinated institutions and long-term planning drive national transformation.',
        'key_points': ['Institutions matter more than personalities.'],
        'topics': [{'title': 'Coordinated Systems', 'points': [{'label': 'Execution', 'description': 'The video links growth to disciplined execution.'}]}],
    })

    response = client.post(
        '/api/analyze',
        data={'url': 'https://www.youtube.com/watch?v=abc123', 'query': 'Summarize'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['success'] is True
    assert payload['result']['topics'] == ['Coordinated Systems']
    assert download_calls['count'] == 0


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
    monkeypatch.setattr('app.background_url_jobs_available', lambda: False)
    monkeypatch.setattr('app.transcribe_audio', lambda _: 'This is a short transcript. It has two sentences.')
    monkeypatch.setattr('app.get_embeddings', lambda chunks: (_ for _ in ()).throw(RuntimeError('skip embeddings')))
    monkeypatch.setattr('app.get_topic_details_from_summary', lambda summary: [
        {'title': 'Topic A', 'explanation': 'Explanation A', 'importance': 'Importance A'},
        {'title': 'Topic B', 'explanation': 'Explanation B', 'importance': 'Importance B'},
    ])

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
    assert payload['result']['topic_details'][0]['title'] == 'Topic A'
    assert payload['result']['articles'] == []


def test_analyze_endpoint_falls_back_when_transcript_is_empty(monkeypatch):
    monkeypatch.setattr('app.background_url_jobs_available', lambda: False)
    monkeypatch.setattr('app.transcribe_audio', lambda _: '')
    monkeypatch.setattr('app.get_topic_details_from_summary', lambda summary: [
        {'title': 'Main topic', 'explanation': 'Explanation', 'importance': 'Importance'}
    ])

    response = client.post(
        '/api/analyze',
        data={'query': 'Summarize'},
        files={'file': ('sample.wav', io.BytesIO(b'fake wav bytes'), 'audio/wav')},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['success'] is True
    assert 'Summarize' in payload['result']['headline']
    assert payload['result']['topics'] == ['Main topic']


def test_analyze_endpoint_generates_articles_when_requested(monkeypatch):
    monkeypatch.setattr('app.background_url_jobs_available', lambda: False)
    monkeypatch.setattr('app.transcribe_audio', lambda _: 'This is a short transcript. It has two sentences.')
    monkeypatch.setattr('app.get_embeddings', lambda chunks: (_ for _ in ()).throw(RuntimeError('skip embeddings')))
    monkeypatch.setattr('app.get_topic_details_from_summary', lambda summary: [
        {'title': 'Topic A', 'explanation': 'Explanation A', 'importance': 'Importance A'}
    ])
    monkeypatch.setattr('app.build_article_package', lambda **kwargs: {
        'topic': kwargs['topic'],
        'article_type': kwargs['article_type'],
        'content': f"Article for {kwargs['topic']}",
        'image_url': 'https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&w=1200&q=80',
        'meta_title': 'Topic A meta title',
        'meta_description': 'Topic A meta description',
        'slug': 'topic-a-meta-title',
        'focus_keyword': kwargs['topic'],
        'secondary_keywords': ['Keyword A'],
        'geo_keywords': ['What is Topic A?'],
        'seo_report': {'seoScore': 8, 'geoScore': 8, 'improvementSuggestions': []},
    })

    response = client.post(
        '/api/analyze',
        data={'query': 'Summarize', 'generate_article': 'true', 'article_count': '1'},
        files={'file': ('sample.wav', io.BytesIO(b'fake wav bytes'), 'audio/wav')},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['success'] is True
    assert payload['result']['articles'][0]['topic'] == 'Topic A'
    assert payload['result']['articles'][0]['content'] == 'Article for Topic A'
    assert payload['result']['articles'][0]['article_type'] == 'Blog Article'
    assert payload['result']['articles'][0]['focus_keyword'] == 'Topic A'


def test_analyze_endpoint_rejects_unsupported_upload_type():
    response = client.post(
        '/api/analyze',
        data={'query': 'Summarize'},
        files={'file': ('sample.txt', io.BytesIO(b'not media'), 'text/plain')},
    )

    assert response.status_code == 400
    assert response.json()['detail'] == 'Unsupported file type. Please upload an audio or video file.'


def test_analyze_endpoint_accepts_pasted_transcript(monkeypatch):
    monkeypatch.setattr('app.background_url_jobs_available', lambda: False)
    monkeypatch.setattr('app.gemini_rag', lambda context, query, source_url='': {
        'heading': 'Transcript-based heading',
        'summary': 'Transcript-based summary.',
        'key_points': ['Point one'],
        'topics': [{'title': 'Transcript Topic', 'points': [{'label': 'Point', 'description': 'Built from pasted transcript.'}]}],
    })

    response = client.post(
        '/api/analyze',
        data={'query': 'Summarize', 'transcript_text': 'This is the pasted transcript. It discusses systems, planning, and execution.'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['success'] is True
    assert payload['result']['headline'] == 'Transcript-based heading'
    assert payload['result']['topics'] == ['Transcript Topic']


def test_analyze_url_source_returns_topics_without_articles(monkeypatch, tmp_path):
    monkeypatch.setattr('app.ANALYSIS_CACHE_DIR', tmp_path / 'analysis-cache')
    monkeypatch.setattr('app.fetch_youtube_metadata', lambda url: {'title': 'Video headline', 'description': 'Video description', 'channel': 'Channel', 'categories': ['News'], 'tags': ['Policy']})
    monkeypatch.setattr('app.direct_gemini_youtube_analysis', lambda url, query, metadata=None: (_ for _ in ()).throw(RuntimeError('unsupported direct analysis')))
    monkeypatch.setattr('app.fetch_youtube_transcript_text', lambda url: LONG_VIDEO_TRANSCRIPT)
    monkeypatch.setattr('app.gemini_rag', lambda context, query, source_url='': {
        'heading': 'Systems, Planning, and National Coordination',
        'summary': 'The video argues that coordinated institutions and long-term planning drive national transformation.',
        'key_points': ['Institutions matter more than personalities.'],
        'topics': [
            {'title': 'Coordinated Systems', 'points': [{'label': 'Execution', 'description': 'The video links growth to disciplined execution.'}]},
            {'title': 'Long-Term Planning', 'points': [{'label': 'Policy continuity', 'description': 'The source contrasts stable planning with policy churn.'}]},
        ],
    })

    result = analyze_url_source('https://www.youtube.com/watch?v=abc123', 'Summarize')

    assert result['topics'] == ['Coordinated Systems', 'Long-Term Planning']
    assert result['articles'] == []


def test_analyze_url_source_downloads_and_transcribes_youtube(monkeypatch, tmp_path):
    monkeypatch.setattr('app.ANALYSIS_CACHE_DIR', tmp_path / 'analysis-cache')
    monkeypatch.setattr('app.fetch_youtube_metadata', lambda url: {'title': 'Video headline', 'description': 'Video description', 'channel': 'Channel', 'categories': ['News'], 'tags': ['Policy']})
    monkeypatch.setattr('app.direct_gemini_youtube_analysis', lambda url, query, metadata=None: (_ for _ in ()).throw(RuntimeError('unsupported direct analysis')))
    monkeypatch.setattr('app.fetch_youtube_transcript_text', lambda url: (_ for _ in ()).throw(RuntimeError('no transcript')))
    monkeypatch.setattr('app.fetch_youtube_subtitles_text', lambda url, output_dir: LONG_VIDEO_TRANSCRIPT)
    monkeypatch.setattr('app.gemini_rag', lambda context, query, source_url='': {
        'heading': 'Systems, Planning, and National Coordination',
        'summary': 'The video argues that coordinated institutions and long-term planning drive national transformation.',
        'key_points': ['Institutions matter more than personalities.'],
        'topics': [{'title': 'Coordinated Systems', 'points': [{'label': 'Execution', 'description': 'The video links growth to disciplined execution.'}]}],
    })

    result = analyze_url_source('https://www.youtube.com/watch?v=abc123', 'Summarize')

    assert result['topics'] == ['Coordinated Systems']
    assert result['articles'] == []


def test_analyze_url_source_uses_real_webpage_content(monkeypatch):
    monkeypatch.setattr('app.extract_webpage_content', lambda url: {
        'title': 'Example Story',
        'meta_description': 'A direct page description.',
        'headings': ['Why This Story Matters'],
        'content': 'This page explains a very specific policy shift, the audience it affects, and the evidence behind the debate.',
    })
    monkeypatch.setattr('app.get_embeddings', lambda chunks: (_ for _ in ()).throw(RuntimeError('skip embeddings')))
    monkeypatch.setattr('app.get_topic_details_from_summary', lambda summary: [
        {'title': 'Policy Shift Impact', 'explanation': 'How the shift affects readers.', 'importance': 'It changes public expectations.'}
    ])

    result = analyze_url_source('https://example.com/story', 'Summarize')

    assert result['source_type'] == 'web-url'
    assert result['topics'] == ['Policy Shift Impact']
    assert result['source_context_preview']


def test_analyze_youtube_source_falls_back_when_direct_analysis_fails(monkeypatch, tmp_path):
    monkeypatch.setattr('app.ANALYSIS_CACHE_DIR', tmp_path / 'analysis-cache')
    monkeypatch.setattr('app.fetch_youtube_metadata', lambda url: {'title': 'Video headline', 'description': 'Video description', 'channel': 'Channel'})
    monkeypatch.setattr('app.direct_gemini_youtube_analysis', lambda url, query, metadata=None: (_ for _ in ()).throw(RuntimeError('direct failed')))
    monkeypatch.setattr('app.fetch_youtube_transcript_text', lambda url: (_ for _ in ()).throw(RuntimeError('no transcript')))
    monkeypatch.setattr('app.fetch_youtube_subtitles_text', lambda url, output_dir: (_ for _ in ()).throw(RuntimeError('no subtitles')))
    monkeypatch.setattr('app.remote_media_fallback_available', lambda: True)
    monkeypatch.setattr('app.download_audio', lambda url, output_dir: 'fake-audio.wav')
    monkeypatch.setattr('app.transcribe_audio', lambda _: 'Recovered transcript from fallback path.')
    monkeypatch.setattr('app.gemini_rag', lambda context, query, source_url='': {
        'heading': 'Recovered analysis',
        'summary': 'Fallback transcription recovered the source content.',
        'key_points': ['Fallback path succeeded.'],
        'topics': [{'title': 'Recovered Topic', 'points': [{'label': 'Recovered point', 'description': 'Recovered transcript from fallback path.'}]}],
    })

    result = analyze_youtube_source('https://www.youtube.com/watch?v=abc123', 'Summarize')

    assert result['topics'] == ['Recovered Topic']
    assert result['direct_analysis'] is False


def test_build_youtube_source_text_does_not_use_description_as_transcript():
    text = build_youtube_source_text(
        {'title': 'Real Title', 'description': 'Follow us on Instagram https://instagram.com/test #news'},
        '',
    )
    assert 'instagram' not in text.lower()
    assert 'https' not in text.lower()
    assert 'follow us' not in text.lower()


def test_analyze_youtube_source_prefers_transcript_first_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr('app.ANALYSIS_CACHE_DIR', tmp_path / 'analysis-cache')
    monkeypatch.setattr('app.ENABLE_DIRECT_GEMINI_YOUTUBE_ANALYSIS', False)
    monkeypatch.setattr('app.fetch_youtube_metadata', lambda url: {'title': 'Video headline', 'description': 'Video description', 'channel': 'Channel'})
    monkeypatch.setattr('app.direct_gemini_youtube_analysis', lambda url, query, metadata=None: (_ for _ in ()).throw(AssertionError('direct analysis should be skipped by default')))
    monkeypatch.setattr('app.fetch_youtube_transcript_text', lambda url: LONG_VIDEO_TRANSCRIPT)
    monkeypatch.setattr('app.gemini_rag', lambda context, query, source_url='': {
        'heading': 'Transcript heading',
        'summary': 'Transcript summary.',
        'key_points': ['Point one'],
        'topics': [{'title': 'Transcript Topic', 'points': [{'label': 'Point', 'description': 'Transcript first worked.'}]}],
    })

    result = analyze_youtube_source('https://www.youtube.com/watch?v=abc123', 'Summarize')

    assert result['heading'] == 'Transcript heading'
    assert result['topics'] == ['Transcript Topic']
    assert result['direct_analysis'] is False


def test_analyze_youtube_source_returns_direct_gemini_result_without_transcript_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr('app.ANALYSIS_CACHE_DIR', tmp_path / 'analysis-cache')
    monkeypatch.setattr('app.ENABLE_DIRECT_GEMINI_YOUTUBE_ANALYSIS', True)
    monkeypatch.setattr('app.fetch_youtube_metadata', lambda url: {'title': 'Video headline', 'description': 'Video description', 'channel': 'Channel'})
    monkeypatch.setattr('app.direct_gemini_youtube_analysis', lambda url, query, metadata=None: {
        'heading': 'Direct analysis heading',
        'summary': 'Direct analysis summary.',
        'key_points': ['Point one'],
        'topics': [{'title': 'Direct Topic', 'points': [{'label': 'Point', 'description': 'Direct path worked.'}]}],
    })
    monkeypatch.setattr('app.fetch_youtube_transcript_text', lambda url: (_ for _ in ()).throw(AssertionError('transcript should not run')))

    result = analyze_youtube_source('https://www.youtube.com/watch?v=abc123', 'Summarize')

    assert result['heading'] == 'Direct analysis heading'
    assert result['topics'] == ['Direct Topic']
    assert result['direct_analysis'] is True


def test_serialize_job_status_promotes_stale_direct_gemini_jobs_to_transcript_extraction(monkeypatch):
    class FakeJob:
        id = 'job-1'
        meta = {
            'stage': 'direct_gemini',
            'message': 'Analyzing video with Gemini...',
            'progress': 12,
            'created_at_ts': 900000,
            'updated_at_ts': 900000,
        }

        def get_status(self, refresh=True):
            return 'started'

    monkeypatch.setattr('app.time.time', lambda: 900000 + YOUTUBE_DIRECT_ANALYSIS_TIMEOUT + 5)
    payload = serialize_job_status(FakeJob())
    assert payload['status'] == 'started'
    assert payload['progress']['stage'] == 'transcript_extraction'
    assert 'Extracting transcript' in payload['progress']['message']


def test_serialize_job_status_marks_overall_job_timeout_as_failed(monkeypatch):
    class FakeJob:
        id = 'job-1'
        meta = {
            'stage': 'audio_fallback',
            'message': 'Captions were not available. Transcribing audio...',
            'progress': 38,
            'created_at_ts': 0,
            'updated_at_ts': 0,
        }

        def get_status(self, refresh=True):
            return 'started'

    monkeypatch.setattr('app.time.time', lambda: ANALYSIS_JOB_MAX_SECONDS + 999)
    payload = serialize_job_status(FakeJob())
    assert payload['status'] == 'failed'
    assert payload['reason'] == 'timeout'


def test_analyze_youtube_endpoint_uses_cache(monkeypatch):
    monkeypatch.setattr('app.background_url_jobs_available', lambda: False)
    calls = {'count': 0}

    def fake_analyze_youtube_source(url, query, progress_callback=None):
        calls['count'] += 1
        return {
            'headline': 'Cached headline',
            'summary': '- point one',
            'topics': ['Topic A'],
            'topic_details': [{'title': 'Topic A', 'explanation': 'Explanation A', 'importance': 'Importance A'}],
            'articles': [],
            'source_cache_key': 'cache-key',
        }

    monkeypatch.setattr('app.analyze_youtube_source', fake_analyze_youtube_source)

    response_one = client.post('/api/analyze-youtube', json={'url': 'https://www.youtube.com/watch?v=abc123', 'query': 'Summarize'})
    response_two = client.post('/api/analyze-youtube', json={'url': 'https://www.youtube.com/watch?v=abc123', 'query': 'Summarize'})

    assert response_one.status_code == 200
    assert response_two.status_code == 200
    assert calls['count'] == 2


def test_generate_article_alias_endpoint(monkeypatch):
    monkeypatch.setattr('app.build_articles_response', lambda payload: {'headline': 'Headline', 'summary': '- point', 'topics': ['Topic A'], 'articles': [{'topic': 'Topic A', 'content': '<h2>Done</h2>'}]})

    response = client.post('/api/generate-article', json={
        'headline': 'Headline',
        'summary': '- point',
        'topics': ['Topic A'],
        'selected_topics': ['Topic A'],
        'article_count': 1,
        'article_type': 'Blog Article',
        'target_audience': 'General readers',
        'source_context': 'Source context',
        'source_cache_key': 'cache-key',
    })

    assert response.status_code == 200
    assert response.json()['result']['articles'][0]['topic'] == 'Topic A'


def test_build_topic_details_bundle_deduplicates_titles(monkeypatch):
    monkeypatch.setattr('app.get_topic_details_from_summary', lambda summary: [
        {'title': 'Topic A', 'explanation': 'One', 'importance': 'Alpha'},
        {'title': 'Topic A', 'explanation': 'Two', 'importance': 'Beta'},
        {'title': 'Topic B', 'explanation': 'Three', 'importance': 'Gamma'},
    ])
    from app import build_topic_details_bundle

    details, _ = build_topic_details_bundle('- point 1')
    assert [item['title'] for item in details] == ['Topic A', 'Topic B']


def test_analyze_endpoint_queues_url_job_when_background_workers_are_available(monkeypatch):
    class FakeJob:
        id = 'job-123'

    monkeypatch.setattr('app.background_url_jobs_available', lambda: True)
    monkeypatch.setattr('app.enqueue_url_analysis', lambda **kwargs: FakeJob())

    response = client.post(
        '/api/analyze',
        data={'url': 'https://www.youtube.com/watch?v=abc123', 'query': 'Summarize'},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload['success'] is True
    assert payload['queued'] is True
    assert payload['jobId'] == 'job-123'
    assert payload['progress']['stage'] == 'transcript_extraction'


def test_job_status_endpoint_returns_completed_result(monkeypatch):
    class FakeJob:
        result = {'headline': 'Ready'}
        meta = {'stage': 'analyzing_topic', 'message': 'Analyzing topic...', 'progress': 84}

        def get_status(self, refresh=True):
            return 'finished'

    monkeypatch.setattr('app.fetch_job', lambda job_id: FakeJob())

    response = client.get('/api/jobs/job-123')

    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'completed'
    assert payload['result']['headline'] == 'Ready'


def test_generate_articles_endpoint_returns_articles(monkeypatch):
    monkeypatch.setattr('app.build_article_package', lambda **kwargs: {
        'topic': kwargs['topic'],
        'article_type': kwargs['article_type'],
        'content': f"Article for {kwargs['topic']}",
        'image_url': 'https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&w=1200&q=80',
        'meta_title': 'Topic A meta title',
        'meta_description': 'Topic A meta description',
        'slug': 'topic-a-meta-title',
        'focus_keyword': kwargs['topic'],
        'secondary_keywords': ['Keyword A'],
        'geo_keywords': ['What is Topic A?'],
        'seo_report': {'seoScore': 8, 'geoScore': 8, 'improvementSuggestions': []},
    })

    response = client.post(
        '/api/articles',
        json={
            'headline': 'Summary headline',
            'summary': '- point 1',
            'topics': ['Topic A'],
            'selected_topics': ['Topic A'],
            'article_count': 1,
            'article_type': 'SEO Article',
            'target_audience': 'Growth teams',
            'source_context': 'A detailed source excerpt.',
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['success'] is True
    assert payload['result']['articles'][0]['content'] == 'Article for Topic A'
    assert payload['result']['articles'][0]['article_type'] == 'SEO Article'


def test_export_article_endpoint_returns_txt_file():
    response = client.post(
        '/api/articles/export',
        json={
            'title': 'My Generated Article',
            'topic': 'Topic A',
            'content_html': '<h2>My Generated Article</h2><p>Body copy</p>',
            'format': 'txt',
        },
    )

    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/plain')
    assert 'attachment; filename="my-generated-article.txt"' == response.headers['content-disposition']
    assert 'My Generated Article' in response.text
    assert 'Body copy' in response.text


def test_map_public_error_message_hides_raw_codes():
    message = map_public_error_message('RuntimeError: upstream failure 502 while contacting provider')
    assert '502' not in message
    assert 'RuntimeError' not in message


def test_map_public_error_message_handles_site_block():
    message = map_public_error_message('That website blocked direct content extraction. Please try another public URL or upload the media file.')
    assert 'blocked automated access' in message


def test_remote_media_fallback_available_when_apify_or_proxy_exists(monkeypatch):
    from app import remote_media_fallback_available

    monkeypatch.setattr('app.ENABLE_REMOTE_MEDIA_DOWNLOAD_FALLBACK', False)
    monkeypatch.setattr('app.APIFY_TOKEN', 'token')
    assert remote_media_fallback_available() is True


def test_fallback_article_avoids_old_template_sections():
    article = fallback_article(
        'Headline Example',
        '- First supporting point.\n- Second supporting point.\n- Third supporting point.\n- Fourth supporting point.',
        'Example Topic',
    )

    assert '<h2>' in article
    assert '<h3>' in article
    assert 'Why This Topic Stands Out' not in article
    assert 'Key Developments' not in article
    assert 'What It Suggests' not in article
    assert '**' not in article
