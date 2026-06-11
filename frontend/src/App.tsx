import React, { useState, useCallback } from 'react'

interface Source {
  title: string
  url: string
}

interface PostInfo {
  author: string
  text: string
  hashtags: string[]
}

type Status = 'idle' | 'extracting' | 'searching' | 'reranking' | 'generating' | 'done' | 'error'

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '0.6rem',
  borderRadius: '8px',
  border: '1px solid #334155',
  background: '#1e293b',
  color: '#e2e8f0',
  fontSize: '0.85rem',
  marginBottom: '0.5rem',
}

export default function App() {
  const [url, setUrl] = useState('https://bsky.app/profile/bsky.app/post/3mnzpiackwk25')
  const [status, setStatus] = useState<Status>('idle')
  const [statusMessage, setStatusMessage] = useState('')
  const [post, setPost] = useState<PostInfo | null>(null)
  const [sources, setSources] = useState<Source[]>([])
  const [bullets, setBullets] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const [modelUsed, setModelUsed] = useState('')

  // API Keys (user provides in frontend)
  const [anthropicKey, setAnthropicKey] = useState('')
  const [openaiKey, setOpenaiKey] = useState('')
  const [tavilyKey, setTavilyKey] = useState('')
  const [showSettings, setShowSettings] = useState(true)
  const [rewriter, setRewriter] = useState('cloud')

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault()
    if (!url.trim()) return
    if (!tavilyKey.trim()) {
      setError('Tavily API Key is required for search.')
      setStatus('error')
      return
    }
    if (!anthropicKey.trim() && !openaiKey.trim()) {
      setError('Provide at least one LLM API key (Anthropic or OpenAI).')
      setStatus('error')
      return
    }

    // Reset state
    setStatus('extracting')
    setStatusMessage('Starting...')
    setPost(null)
    setSources([])
    setBullets([])
    setError(null)
    setModelUsed('')

    try {
      const response = await fetch('/api/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url: url.trim(),
          rewriter,
          api_keys: {
            anthropic: anthropicKey || null,
            openai: openaiKey || null,
            tavily: tavilyKey || null,
          },
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error('No response body')

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        let eventType = ''
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7)
          } else if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6))
            switch (eventType) {
              case 'status':
                setStatus(data.step)
                setStatusMessage(data.message)
                break
              case 'post':
                setPost(data)
                break
              case 'sources':
                setSources(data.sources)
                break
              case 'bullet':
                setBullets(prev => [...prev, data.bullet])
                break
              case 'done':
                setStatus('done')
                setModelUsed(data.model_used)
                break
              case 'error':
                setStatus('error')
                setError(data.error)
                break
            }
          }
        }
      }
    } catch (err) {
      setStatus('error')
      setError(err instanceof Error ? err.message : 'Unknown error')
    }
  }, [url, anthropicKey, openaiKey, tavilyKey, rewriter])

  const isLoading = !['idle', 'done', 'error'].includes(status)

  return (
    <div>
      <h1 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>
        🔍 Contextual Post Explainer
      </h1>
      <p style={{ color: '#94a3b8', marginBottom: '1.5rem', fontSize: '0.9rem' }}>
        Paste a Bluesky post URL to get AI-powered context explanation
      </p>

      {/* API Keys Section */}
      <div style={{ marginBottom: '1rem', padding: '1rem', background: '#1e293b', borderRadius: '8px', border: '1px solid #334155' }}>
        <div
          style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
          onClick={() => setShowSettings(!showSettings)}
        >
          <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>⚙️ API Keys</span>
          <span style={{ fontSize: '0.75rem', color: '#64748b' }}>{showSettings ? '▼' : '▶'}</span>
        </div>
        {showSettings && (
          <div style={{ marginTop: '0.75rem' }}>
            <label htmlFor="tavily-key" style={{ display: 'block', fontSize: '0.75rem', color: '#64748b', marginBottom: '0.2rem' }}>
              Tavily API Key (required for search)
            </label>
            <input
              id="tavily-key"
              type="password"
              value={tavilyKey}
              onChange={e => setTavilyKey(e.target.value)}
              placeholder="tvly-..."
              style={inputStyle}
            />
            <label htmlFor="anthropic-key" style={{ display: 'block', fontSize: '0.75rem', color: '#64748b', marginBottom: '0.2rem' }}>
              Anthropic API Key (for Claude)
            </label>
            <input
              id="anthropic-key"
              type="password"
              value={anthropicKey}
              onChange={e => setAnthropicKey(e.target.value)}
              placeholder="sk-ant-..."
              style={inputStyle}
            />
            <label htmlFor="openai-key" style={{ display: 'block', fontSize: '0.75rem', color: '#64748b', marginBottom: '0.2rem' }}>
              OpenAI API Key (for GPT-4o, optional)
            </label>
            <input
              id="openai-key"
              type="password"
              value={openaiKey}
              onChange={e => setOpenaiKey(e.target.value)}
              placeholder="sk-..."
              style={inputStyle}
            />
            <p style={{ fontSize: '0.7rem', color: '#475569', margin: '0.25rem 0 0 0' }}>
              The model is chosen automatically based on which key you provide. Keys are sent per-request only.
            </p>
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} style={{ marginBottom: '1.5rem' }}>
        <label htmlFor="post-url" style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.85rem', color: '#94a3b8' }}>
          Post URL
        </label>
        <input
          id="post-url"
          type="text"
          value={url}
          onChange={e => setUrl(e.target.value)}
          placeholder="https://bsky.app/profile/user.bsky.social/post/..."
          disabled={isLoading}
          style={{
            width: '100%',
            padding: '0.75rem',
            borderRadius: '8px',
            border: '1px solid #334155',
            background: '#1e293b',
            color: '#e2e8f0',
            fontSize: '0.95rem',
            marginBottom: '0.75rem',
          }}
        />
        <label htmlFor="rewriter-select" style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.85rem', color: '#94a3b8' }}>
          Query Rewriter
        </label>
        <select
          id="rewriter-select"
          value={rewriter}
          onChange={e => setRewriter(e.target.value)}
          disabled={isLoading}
          aria-label="Query Rewriter"
          style={{
            width: '100%',
            padding: '0.6rem',
            borderRadius: '8px',
            border: '1px solid #334155',
            background: '#1e293b',
            color: '#e2e8f0',
            fontSize: '0.85rem',
            marginBottom: '0.75rem',
          }}
        >
          <option value="cloud">Cloud LLM (higher quality, uses tokens)</option>
          <option value="local">Local CPU — Qwen2.5-0.5B (free, slower)</option>
        </select>
        <button
          type="submit"
          disabled={isLoading || !url.trim()}
          style={{
            width: '100%',
            padding: '0.7rem 1.5rem',
            borderRadius: '8px',
            border: 'none',
            background: isLoading ? '#475569' : '#3b82f6',
            color: '#fff',
            fontSize: '0.95rem',
            cursor: isLoading ? 'not-allowed' : 'pointer',
          }}
        >
          {isLoading ? statusMessage : 'Explain'}
        </button>
      </form>

      {error && (
        <div role="alert" style={{ padding: '1rem', background: '#7f1d1d33', borderRadius: '8px', marginBottom: '1rem', border: '1px solid #991b1b' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {post && (
        <div style={{ padding: '1rem', background: '#1e293b', borderRadius: '8px', marginBottom: '1rem', border: '1px solid #334155' }}>
          <div style={{ fontSize: '0.85rem', color: '#94a3b8' }}>@{post.author}</div>
          <div style={{ marginTop: '0.3rem' }}>{post.text}</div>
          {post.hashtags.length > 0 && (
            <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: '#60a5fa' }}>
              {post.hashtags.map(h => `#${h}`).join(' ')}
            </div>
          )}
        </div>
      )}

      {sources.length > 0 && (
        <div style={{ marginBottom: '1rem' }}>
          <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.3rem' }}>Sources found:</div>
          {sources.map((s, i) => (
            <div key={i} style={{ fontSize: '0.8rem', color: '#60a5fa', marginLeft: '0.5rem' }}>
              <a href={s.url} target="_blank" rel="noopener noreferrer" style={{ color: '#60a5fa' }}>{s.title}</a>
            </div>
          ))}
        </div>
      )}

      {bullets.length > 0 && (
        <div style={{ padding: '1rem', background: '#1e293b', borderRadius: '8px', border: '1px solid #334155' }}>
          <div style={{ fontSize: '0.85rem', color: '#94a3b8', marginBottom: '0.5rem' }}>Explanation:</div>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {bullets.map((b, i) => (
              <li key={i} style={{ marginBottom: '0.5rem', paddingLeft: '1rem', position: 'relative' }}>
                <span style={{ position: 'absolute', left: 0 }} aria-hidden="true">•</span>
                {b}
              </li>
            ))}
          </ul>
          {modelUsed && (
            <div style={{ marginTop: '0.75rem', fontSize: '0.75rem', color: '#64748b' }}>
              Model: {modelUsed}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
