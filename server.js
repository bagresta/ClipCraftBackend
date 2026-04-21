const express = require('express');
const multer = require('multer');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { v4: uuidv4 } = require('uuid');
const OpenAI = require('openai');
const ffmpeg = require('fluent-ffmpeg');
const ffmpegStatic = require('ffmpeg-static');
const ffprobeStatic = require('ffprobe-static');

ffmpeg.setFfmpegPath(ffmpegStatic);
ffmpeg.setFfprobePath(ffprobeStatic.path);
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;
const BASE_URL = process.env.BASE_URL || `http://localhost:${PORT}`;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

const uploadsDir = path.join(__dirname, 'uploads');
const jobsDir = path.join(__dirname, 'jobs');
const outputDir = path.join(__dirname, 'outputs');
[uploadsDir, jobsDir, outputDir].forEach(dir => {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
});

const storage = multer.diskStorage({
  destination: uploadsDir,
  filename: (req, file, cb) => cb(null, `${Date.now()}-${Math.random().toString(36).substring(7)}-${file.originalname}`),
});
const upload = multer({ storage, limits: { fileSize: 500 * 1024 * 1024 } });
const jobs = new Map();

function pad(num, digits = 2) { return String(num).padStart(digits, '0'); }
function formatSRTTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.round((seconds % 1) * 1000);
  return `${pad(h)}:${pad(m)}:${pad(s)},${pad(ms, 3)}`;
}

function getVideoInfo(filePath) {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(filePath, (err, metadata) => {
      if (err) return reject(err);
      const stream = metadata.streams.find(s => s.codec_type === 'video') || metadata.streams[0];
      resolve({ duration: metadata.format.duration || 0, width: stream?.width || 1080, height: stream?.height || 1920 });
    });
  });
}

// Memory-efficient FFmpeg options
const FFMPEG_OPTS = ['-threads', '1', '-preset', 'ultrafast'];

async function normalizeClip(filePath, outputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(filePath)
      .output(outputPath)
      .videoCodec('libx264')
      .audioCodec('aac')
      .outputOptions([...FFMPEG_OPTS, '-crf', '28', '-vf', 'scale=720:-2', '-movflags', '+faststart'])
      .on('error', reject)
      .on('end', resolve)
      .run();
  });
}

async function photoToVideo(filePath, outputPath, duration = 5) {
  return new Promise((resolve, reject) => {
    ffmpeg()
      .input(filePath)
      .inputOptions(['-loop', '1', '-t', String(duration)])
      .output(outputPath)
      .videoCodec('libx264')
      .outputOptions([...FFMPEG_OPTS, '-crf', '28', '-vf', 'scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2', '-movflags', '+faststart'])
      .on('error', reject)
      .on('end', resolve)
      .run();
  });
}

async function transcribeClip(filePath, openaiKey) {
  const client = new OpenAI({ apiKey: openaiKey });
  try {
    const audioPath = filePath.replace(/\.[^.]+$/, '_audio.mp3');
    await new Promise((resolve, reject) => {
      ffmpeg(filePath)
        .output(audioPath)
        .outputOptions(['-threads', '1', '-q:a', '5', '-map', 'a'])
        .on('error', reject)
        .on('end', resolve)
        .run();
    });
    if (!fs.existsSync(audioPath) || fs.statSync(audioPath).size < 2048) {
      try { fs.unlinkSync(audioPath); } catch(_) {}
      return { segments: [], text: '' };
    }
    const audioFile = fs.createReadStream(audioPath);
    const transcript = await client.audio.transcriptions.create({
      file: audioFile, model: 'whisper-1', response_format: 'verbose_json', timestamp_granularity: 'segment',
    });
    try { fs.unlinkSync(audioPath); } catch(_) {}
    return { segments: transcript.segments || [], text: transcript.text || '' };
  } catch (error) {
    console.error('Transcription error:', error.message);
    return { segments: [], text: '' };
  }
}

function detectSilence(filePath) {
  return new Promise((resolve) => {
    const silenceRanges = [];
    let currentStart = null;
    const child = spawn(ffmpegStatic, ['-i', filePath, '-af', 'silencedetect=noise=-35dB:d=1.5', '-f', 'null', '-'], { stdio: ['ignore', 'pipe', 'pipe'] });
    let stderr = '';
    child.stderr.on('data', d => stderr += d.toString());
    child.on('close', () => {
      for (const line of stderr.split('\n')) {
        const s = line.match(/silence_start:\s*([\d.]+)/);
        const e = line.match(/silence_end:\s*([\d.]+)/);
        if (s) currentStart = parseFloat(s[1]);
        if (e && currentStart !== null) { silenceRanges.push({ start: currentStart, end: parseFloat(e[1]) }); currentStart = null; }
      }
      resolve(silenceRanges);
    });
    child.on('error', () => resolve([]));
  });
}

function invertSilenceRanges(silenceRanges, totalDuration) {
  const keepRanges = [];
  let pos = 0;
  for (const s of silenceRanges) {
    const ps = Math.max(0, s.start - 0.3);
    const pe = Math.min(totalDuration, s.end + 0.3);
    if (pos < ps) keepRanges.push({ start: pos, end: ps });
    pos = pe;
  }
  if (pos < totalDuration) keepRanges.push({ start: pos, end: totalDuration });
  return keepRanges;
}

async function trimClip(inputPath, outputPath, start, duration) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .seekInput(start)
      .duration(duration)
      .output(outputPath)
      .videoCodec('libx264')
      .audioCodec('aac')
      .outputOptions([...FFMPEG_OPTS, '-crf', '28'])
      .on('error', reject)
      .on('end', resolve)
      .run();
  });
}

async function concatClips(concatFilePath, outputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg()
      .input(concatFilePath)
      .inputOptions(['-f', 'concat', '-safe', '0'])
      .output(outputPath)
      .videoCodec('libx264')
      .audioCodec('aac')
      .outputOptions([...FFMPEG_OPTS, '-crf', '28'])
      .on('error', reject)
      .on('end', resolve)
      .run();
  });
}

async function scaleAndPad(inputPath, outputPath, width, height) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .output(outputPath)
      .videoCodec('libx264')
      .audioCodec('aac')
      .outputOptions([...FFMPEG_OPTS, '-crf', '28'])
      .videoFilters(`scale=${width}:${height}:force_original_aspect_ratio=decrease,pad=${width}:${height}:(ow-iw)/2:(oh-ih)/2`)
      .on('error', reject)
      .on('end', resolve)
      .run();
  });
}

async function processJob(jobId) {
  const job = jobs.get(jobId);
  if (!job) return;
  try {
    const jobDir = path.join(jobsDir, jobId);
    if (!fs.existsSync(jobDir)) fs.mkdirSync(jobDir, { recursive: true });
    const openaiKey = job.openaiKey || OPENAI_API_KEY;
    if (!openaiKey) throw new Error('No OpenAI API key provided');

    // Stage 1: Normalize
    job.stage = 'normalizing'; job.status = 'processing';
    const normalizedClips = [];
    for (let i = 0; i < job.fileIds.length; i++) {
      job.progress = 5 + Math.round((i / job.fileIds.length) * 15);
      job.message = `Normalizing clip ${i + 1} of ${job.fileIds.length}...`;
      const fileId = job.fileIds[i];
      const filePath = path.join(uploadsDir, fileId);
      if (!fs.existsSync(filePath)) throw new Error(`File not found: ${fileId}`);
      const isImage = ['.jpg','.jpeg','.png','.heic','.webp'].some(e => fileId.toLowerCase().endsWith(e));
      const normalizedPath = path.join(jobDir, `norm_${i}.mp4`);
      if (isImage) await photoToVideo(filePath, normalizedPath);
      else await normalizeClip(filePath, normalizedPath);
      const info = await getVideoInfo(normalizedPath);
      normalizedClips.push({ index: i, path: normalizedPath, duration: info.duration });
      // Delete original upload to free space
      try { fs.unlinkSync(filePath); } catch(_) {}
    }

    // Stage 2: Transcribe
    job.stage = 'transcribing';
    const clipTranscripts = [];
    for (let i = 0; i < normalizedClips.length; i++) {
      job.progress = 20 + Math.round((i / normalizedClips.length) * 15);
      job.message = `Transcribing clip ${i + 1}...`;
      const t = await transcribeClip(normalizedClips[i].path, openaiKey);
      clipTranscripts.push({ index: i, text: t.text, segments: t.segments });
    }

    // Stage 3: Silence detection
    job.stage = 'analyzing';
    const clipKeepRanges = [];
    for (let i = 0; i < normalizedClips.length; i++) {
      job.progress = 35 + Math.round((i / normalizedClips.length) * 15);
      job.message = `Analyzing clip ${i + 1}...`;
      const silenceRanges = await detectSilence(normalizedClips[i].path);
      const keepRanges = invertSilenceRanges(silenceRanges, normalizedClips[i].duration);
      clipKeepRanges.push({ index: i, keepRanges: keepRanges.length > 0 ? keepRanges : [{ start: 0, end: normalizedClips[i].duration }] });
    }

    // Stage 4: AI plan
    job.stage = 'planning'; job.progress = 50; job.message = 'Planning edits...';
    let editPlan = { editOrder: normalizedClips.map((_, i) => i), commentary: '', title: 'ClipCraft Video', hashtags: ['clipcraft'], description: 'Created with ClipCraft' };
    try {
      const client = new OpenAI({ apiKey: openaiKey });
      const prompt = clipTranscripts.map((t, i) => `Clip ${i}: "${t.text || '(no speech)'}"`).join('\n');
      const res = await client.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [
          { role: 'system', content: `Video editing AI. Style: ${job.style || 'engaging'}. Return JSON only, no markdown.` },
          { role: 'user', content: `${prompt}\n\nReturn: {"editOrder":[0,1...],"commentary":"narration","title":"title","hashtags":["tag"],"description":"desc"}` },
        ],
        max_tokens: 500,
      });
      const content = res.choices[0].message.content.replace(/```json\n?/g,'').replace(/```\n?/g,'').trim();
      editPlan = JSON.parse(content);
    } catch (e) { console.warn('AI plan failed:', e.message); }

    // Stage 5: TTS
    job.stage = 'generating_tts'; job.progress = 60; job.message = 'Generating voiceover...';
    let voiceoverPath = null;
    if (editPlan.commentary) {
      try {
        const client = new OpenAI({ apiKey: openaiKey });
        const mp3Path = path.join(jobDir, 'voiceover.mp3');
        const res = await client.audio.speech.create({ model: 'tts-1', voice: 'nova', input: editPlan.commentary.substring(0, 4096) });
        fs.writeFileSync(mp3Path, Buffer.from(await res.arrayBuffer()));
        voiceoverPath = mp3Path;
      } catch (e) { console.warn('TTS failed:', e.message); }
    }

    // Stage 6: Render each aspect ratio
    job.stage = 'rendering';
    const aspectConfigs = { '9:16': { w: 720, h: 1280 }, '16:9': { w: 1280, h: 720 }, '1:1': { w: 720, h: 720 }, '4:5': { w: 720, h: 900 } };
    const outputUrls = {};

    for (let ai = 0; ai < job.aspectRatios.length; ai++) {
      const ar = job.aspectRatios[ai];
      const cfg = aspectConfigs[ar] || { w: 720, h: 1280 };
      job.progress = 65 + Math.round((ai / job.aspectRatios.length) * 30);
      job.message = `Rendering ${ar}...`;

      // Trim boring bits and build concat list
      const concatLines = [];
      for (const clipIdx of (editPlan.editOrder || normalizedClips.map((_,i)=>i))) {
        if (clipIdx >= normalizedClips.length) continue;
        const clip = normalizedClips[clipIdx];
        const keepRanges = clipKeepRanges[clipIdx]?.keepRanges || [{ start: 0, end: clip.duration }];
        for (let ri = 0; ri < keepRanges.length; ri++) {
          const range = keepRanges[ri];
          const dur = range.end - range.start;
          if (dur < 0.5) continue;
          const trimPath = path.join(jobDir, `trim_${clipIdx}_${ri}_${ar.replace(':','_')}.mp4`);
          await trimClip(clip.path, trimPath, range.start, dur);
          concatLines.push(`file '${trimPath.replace(/\\/g, '/')}'`);
        }
      }

      if (concatLines.length === 0) throw new Error('No video segments after trimming');

      const concatFile = path.join(jobDir, `concat_${ar.replace(':','_')}.txt`);
      fs.writeFileSync(concatFile, concatLines.join('\n'));

      const concatedPath = path.join(jobDir, `concat_out_${ar.replace(':','_')}.mp4`);
      await concatClips(concatFile, concatedPath);

      const scaledPath = path.join(jobDir, `scaled_${ar.replace(':','_')}.mp4`);
      await scaleAndPad(concatedPath, scaledPath, cfg.w, cfg.h);
      try { fs.unlinkSync(concatedPath); } catch(_) {}

      // Mix voiceover if available
      const finalPath = path.join(outputDir, `${jobId}_${ar.replace(':','_')}.mp4`);
      if (voiceoverPath) {
        await new Promise((resolve, reject) => {
          ffmpeg()
            .input(scaledPath).input(voiceoverPath)
            .complexFilter('[0:a][1:a]amix=inputs=2:duration=first[a]')
            .map('0:v').map('[a]')
            .output(finalPath)
            .videoCodec('copy').audioCodec('aac')
            .outputOptions(['-threads', '1'])
            .on('error', reject).on('end', resolve).run();
        });
      } else {
        fs.copyFileSync(scaledPath, finalPath);
      }
      try { fs.unlinkSync(scaledPath); } catch(_) {}
      outputUrls[ar] = `${BASE_URL}/jobs/${jobId}/download/${ar.replace(':', '_')}`;
    }

    // Clean up normalized clips
    for (const clip of normalizedClips) { try { fs.unlinkSync(clip.path); } catch(_) {} }

    job.status = 'complete'; job.progress = 100; job.stage = 'done'; job.message = 'Done!';
    job.outputUrls = outputUrls;
    job.commentary = { title: editPlan.title, description: editPlan.description, hashtags: editPlan.hashtags, commentary: editPlan.commentary };

    setTimeout(() => {
      try { fs.rmSync(path.join(jobsDir, jobId), { recursive: true, force: true }); jobs.delete(jobId); } catch(_) {}
    }, 3600000);
  } catch (error) {
    console.error('Job failed:', error);
    job.status = 'failed'; job.error = error.message; job.message = error.message;
  }
}

app.use(cors());
app.use(express.json());
app.get('/health', (req, res) => res.json({ status: 'ok', version: '1.0.0' }));

app.post('/upload', upload.array('files'), (req, res) => {
  if (!req.files?.length) return res.status(400).json({ error: 'No files uploaded' });
  res.json(req.files.map(f => ({ id: f.filename, originalname: f.originalname, mimetype: f.mimetype })));
});

app.post('/jobs', (req, res) => {
  const { fileIds, style, aspectRatios, maxDuration, openaiKey } = req.body;
  if (!fileIds?.length) return res.status(400).json({ error: 'fileIds required' });
  if (!aspectRatios?.length) return res.status(400).json({ error: 'aspectRatios required' });
  const jobId = uuidv4();
  jobs.set(jobId, { id: jobId, fileIds, style: style || 'engaging', aspectRatios, maxDuration: maxDuration || 0, openaiKey, status: 'queued', progress: 0, stage: 'queued', message: 'Queued...', outputUrls: {} });
  setImmediate(() => processJob(jobId));
  res.json({ jobId });
});

app.get('/jobs/:id', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: 'Job not found' });
  res.json({ id: job.id, status: job.status, progress: job.progress, stage: job.stage, message: job.message, outputUrls: job.outputUrls, commentary: job.commentary, error: job.error });
});

app.get('/jobs/:jobId/download/:aspect', (req, res) => {
  const { jobId, aspect } = req.params;
  if (!jobs.get(jobId)) return res.status(404).json({ error: 'Job not found' });
  const filePath = path.join(outputDir, `${jobId}_${aspect}.mp4`);
  if (!fs.existsSync(filePath)) return res.status(404).json({ error: 'File not found' });
  res.download(filePath, `clipcraft_${aspect}.mp4`);
});

app.listen(PORT, () => console.log(`ClipCraft backend on port ${PORT}`));