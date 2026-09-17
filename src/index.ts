import 'dotenv/config';
import fs from 'fs';
import os from 'os';
import path from 'path';
import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import multer from 'multer';
import OpenAI from 'openai';   // Groq uses the same OpenAI-compatible SDK

// ── Inline web chat UI (served at GET /) ──────────────────────────────────────
const WEB_UI = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1"/>
<title>AI Assistant</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; font-family: -apple-system, "Segoe UI", system-ui, sans-serif; background: #fff; color: #202124; }
  body { display: flex; flex-direction: column; height: 100dvh; }

  /* Header */
  #header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 20px; background: #fff;
    border-bottom: 1px solid #f1f3f4; flex-shrink: 0;
  }
  #header-left { display: flex; align-items: center; gap: 10px; }
  #logo {
    width: 34px; height: 34px; border-radius: 50%;
    background: #e8f0fe; display: flex; align-items: center;
    justify-content: center; font-size: 16px; color: #4285f4; flex-shrink: 0;
  }
  #header h1 { font-size: 17px; font-weight: 600; color: #202124; letter-spacing: -0.2px; }
  #status { font-size: 12px; font-weight: 500; color: #34a853; }

  /* Messages */
  #messages {
    flex: 1; overflow-y: auto; padding: 20px 16px 12px;
    display: flex; flex-direction: column; gap: 20px;
  }

  /* Bot message — Gemini style: no bubble, plain text with avatar */
  .msg-bot-row { display: flex; align-items: flex-start; gap: 12px; padding-right: 24px; }
  .bot-avatar {
    width: 30px; height: 30px; border-radius: 50%; background: #e8f0fe;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; color: #4285f4; flex-shrink: 0; margin-top: 1px;
  }
  .msg-bot-text { font-size: 15px; line-height: 1.6; color: #202124; white-space: pre-wrap; word-break: break-word; }
  .msg-typing { font-size: 14px; color: #9aa0a6; font-style: italic; }

  /* User message — ChatGPT pill bubble */
  .msg-user-row { display: flex; justify-content: flex-end; }
  .msg-user-bubble {
    background: #f1f3f4; border-radius: 20px;
    padding: 10px 16px; max-width: 80%;
    font-size: 15px; line-height: 1.5; color: #202124;
    white-space: pre-wrap; word-break: break-word;
  }

  /* Input area */
  #inputarea {
    padding: 8px 12px 10px; background: #fff;
    border-top: 1px solid #f1f3f4; flex-shrink: 0;
  }
  #inputbar {
    display: flex; align-items: flex-end; gap: 8px;
    background: #f1f3f4; border-radius: 26px;
    padding: 8px 16px;
  }
  #textinput {
    flex: 1; border: none; background: transparent; outline: none;
    font-size: 15px; color: #202124; resize: none; max-height: 120px;
    font-family: inherit; line-height: 1.5; padding: 4px 0;
  }
  #textinput::placeholder { color: #9aa0a6; }
  #actions { display: flex; align-items: flex-end; gap: 6px; padding-bottom: 2px; }
  .btn {
    width: 34px; height: 34px; border-radius: 50%; border: none;
    cursor: pointer; font-size: 16px; display: flex; align-items: center;
    justify-content: center; flex-shrink: 0; transition: background 0.15s;
  }
  #sendbtn  { background: #4285f4; color: #fff; }
  #sendbtn:hover { background: #1a73e8; }
  #sendbtn:disabled { background: #dadce0; cursor: default; }
  #micbtn   { background: #e8eaed; color: #202124; }
  #micbtn:hover { background: #dadce0; }
  #micbtn.recording { background: #ea4335; color: #fff; animation: pulse 1s infinite; }
  #stopbtn  { background: #e8eaed; color: #202124; display: none; }
  #stopbtn:hover { background: #dadce0; }
  #disclaimer { text-align: center; font-size: 11px; color: #9aa0a6; margin-top: 6px; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.6} }
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <div id="logo">✦</div>
    <h1>AI Assistant</h1>
  </div>
  <span id="status">● Ready</span>
</div>

<div id="messages"></div>

<div id="inputarea">
  <div id="inputbar">
    <textarea id="textinput" rows="1" placeholder="Ask anything…" autocomplete="off"></textarea>
    <div id="actions">
      <button class="btn" id="stopbtn" title="Stop speaking">🔇</button>
      <button class="btn" id="micbtn" title="Voice input">🎙</button>
      <button class="btn" id="sendbtn" title="Send" disabled>↑</button>
    </div>
  </div>
  <p id="disclaimer">AI can make mistakes. Check important info.</p>
</div>

<script>
  const messagesEl = document.getElementById('messages');
  const textInput  = document.getElementById('textinput');
  const sendBtn    = document.getElementById('sendbtn');
  const micBtn     = document.getElementById('micbtn');
  const stopBtn    = document.getElementById('stopbtn');
  const statusEl   = document.getElementById('status');

  let history = [];
  let mediaRecorder = null;
  let audioChunks = [];

  // ── Helpers ──────────────────────────────────────────────────────────────
  function setStatus(text, color) {
    statusEl.textContent = text;
    statusEl.style.color = color || '#34a853';
  }

  function appendBotMessage(text, isTyping) {
    const row = document.createElement('div');
    row.className = 'msg-bot-row';
    row.innerHTML =
      '<div class="bot-avatar">✦</div>' +
      '<div class="' + (isTyping ? 'msg-bot-text msg-typing' : 'msg-bot-text') + '">' +
      escHtml(text) + '</div>';
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return row;
  }

  function appendUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'msg-user-row';
    row.innerHTML = '<div class="msg-user-bubble">' + escHtml(text) + '</div>';
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function escHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  function removeTyping() {
    const t = messagesEl.querySelector('.msg-typing');
    if (t) t.closest('.msg-bot-row').remove();
  }

  // ── Auto-resize textarea ─────────────────────────────────────────────────
  textInput.addEventListener('input', () => {
    textInput.style.height = 'auto';
    textInput.style.height = Math.min(textInput.scrollHeight, 120) + 'px';
    sendBtn.disabled = !textInput.value.trim();
  });

  // ── Text-to-speech ───────────────────────────────────────────────────────
  function speak(text) {
    if (!window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.lang = 'en-US';
    utt.onstart  = () => { stopBtn.style.display = 'flex'; setStatus('🔊 Speaking…', '#8ab4f8'); };
    utt.onend    = () => { stopBtn.style.display = 'none';  setStatus('● Ready'); };
    utt.onerror  = () => { stopBtn.style.display = 'none';  setStatus('● Ready'); };
    window.speechSynthesis.speak(utt);
  }

  stopBtn.addEventListener('click', () => {
    window.speechSynthesis?.cancel();
    stopBtn.style.display = 'none';
    setStatus('● Ready');
  });

  // ── Send message ─────────────────────────────────────────────────────────
  async function sendChat(text) {
    if (!text.trim()) return;
    textInput.value = '';
    textInput.style.height = 'auto';
    sendBtn.disabled = true;
    appendUserMessage(text);
    history.push({ role: 'user', content: text });

    const typingRow = appendBotMessage('Thinking…', true);
    setStatus('● Thinking…', '#9aa0a6');

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: history }),
      });
      const data = await res.json();
      const reply = data.reply ?? data.error ?? 'No response';
      removeTyping();
      appendBotMessage(reply, false);
      history.push({ role: 'assistant', content: reply });
      speak(reply);
    } catch (e) {
      removeTyping();
      appendBotMessage('⚠️ Could not reach the server.', false);
    } finally {
      setStatus('● Ready');
    }
  }

  sendBtn.addEventListener('click', () => sendChat(textInput.value));
  textInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(textInput.value); }
  });

  // ── Voice input ───────────────────────────────────────────────────────────
  micBtn.addEventListener('click', async () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
      mediaRecorder.onstop = async () => {
        micBtn.classList.remove('recording');
        setStatus('● Transcribing…', '#9aa0a6');
        stream.getTracks().forEach(t => t.stop());
        const blob = new Blob(audioChunks, { type: 'audio/webm' });
        const fd = new FormData();
        fd.append('audio', blob, 'recording.webm');
        try {
          const res = await fetch('/transcribe', { method: 'POST', body: fd });
          const data = await res.json();
          if (data.text) { textInput.value = data.text; sendBtn.disabled = false; await sendChat(data.text); }
          else setStatus('● Could not transcribe', '#ea4335');
        } catch { setStatus('● Transcription failed', '#ea4335'); }
      };
      mediaRecorder.start();
      micBtn.classList.add('recording');
      setStatus('⏺ Listening… tap again to stop', '#ea4335');
    } catch (e) { setStatus('● Mic access denied', '#ea4335'); }
  });

  // ── Welcome ───────────────────────────────────────────────────────────────
  appendBotMessage('Hello! How can I help you today?', false);
</script>
</body>
</html>`;

// ── Config ────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3001;
const CORS_ORIGIN = process.env.CORS_ORIGIN ?? '*';
const GROQ_MODEL = process.env.GROQ_MODEL ?? 'llama-3.3-70b-versatile';
const SYSTEM_PROMPT =
  process.env.SYSTEM_PROMPT ??
  'You are a helpful AI assistant. Be concise, friendly, and accurate.';

if (!process.env.GROQ_API_KEY) {
  console.error('❌  GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.');
  console.error('    Get a FREE key at https://console.groq.com');
  process.exit(1);
}

// ── Groq client (uses OpenAI-compatible SDK, free tier) ───────────────────────
const groq = new OpenAI({
  apiKey: process.env.GROQ_API_KEY,
  baseURL: 'https://api.groq.com/openai/v1',
});

// ── Types ─────────────────────────────────────────────────────────────────────
type Role = 'user' | 'assistant';

interface ChatRequestBody {
  messages: { role: Role; content: string }[];
}

// ── Express app ───────────────────────────────────────────────────────────────
const app = express();

// multer stores the uploaded audio in a temp dir; we clean it up after use
const upload = multer({ dest: os.tmpdir() });

app.use(cors({ origin: CORS_ORIGIN }));
app.use(express.json());

// Web chat UI — open in any browser (phone or desktop)
app.get('/', (_req, res) => {
  res.setHeader('Content-Type', 'text/html');
  res.send(WEB_UI);
});

// Health check
app.get('/health', (_req, res) => {
  res.json({ status: 'ok' });
});

// Chat endpoint
app.post('/chat', async (req: Request, res: Response, next: NextFunction) => {
  const body = req.body as ChatRequestBody;

  if (!Array.isArray(body?.messages) || body.messages.length === 0) {
    res.status(400).json({ error: 'messages array is required and must not be empty.' });
    return;
  }

  // Validate each message has the expected shape
  for (const msg of body.messages) {
    if (!msg.role || !msg.content || !['user', 'assistant'].includes(msg.role)) {
      res.status(400).json({ error: 'Each message must have a valid role and content.' });
      return;
    }
  }

  try {
    const completion = await groq.chat.completions.create({
      model: GROQ_MODEL,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        ...body.messages,
      ],
    });

    const reply = completion.choices[0]?.message?.content ?? '(no response)';
    res.json({ reply });
  } catch (err) {
    next(err);
  }
});

// Transcribe endpoint — uses Groq's free Whisper endpoint
app.post(
  '/transcribe',
  upload.single('audio'),
  async (req: Request, res: Response, next: NextFunction) => {
    if (!req.file) {
      res.status(400).json({ error: 'No audio file uploaded.' });
      return;
    }

    const tmpPath = req.file.path;
    // Preserve the original extension so Whisper knows the format.
    // Mobile (expo-av) sends audio/mp4; web browser sends audio/webm.
    const mime = req.file.mimetype || 'audio/mp4';
    const ext = mime.includes('webm') ? '.webm' : mime.includes('mp4') ? '.mp4' : '.wav';
    const audioPath = tmpPath + ext;

    try {
      fs.renameSync(tmpPath, audioPath);

      const transcription = await groq.audio.transcriptions.create({
        file: fs.createReadStream(audioPath),
        model: 'whisper-large-v3-turbo',
      });

      res.json({ text: transcription.text });
    } catch (err) {
      next(err);
    } finally {
      try { fs.unlinkSync(audioPath); } catch { /* ignore */ }
    }
  },
);

// Global error handler
app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
  console.error('Unhandled error:', err.message);
  res.status(500).json({ error: 'Internal server error.' });
});

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, '0.0.0.0', () => {
  console.log(`✅  Backend listening on http://localhost:${PORT}`);
  console.log(`📱  Open on your phone: http://<YOUR-LAN-IP>:${PORT}`);
  console.log(`    (run 'ipconfig' on Windows to find your LAN IP)`);
});
