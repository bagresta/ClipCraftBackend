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
[uploadsDir, jobsDir, outputDir].forEach(d => fs.mkdirSync(d, { recursive: true }));

const upload = multer({
  dest: uploadsDir,
  limits: { fileSize: 300 * 1024 * 1024 },
});

const jobs = new Map();

// ── helpers ──────────────────────────────────────────────────────────────────

function run(cmd) {
  return new Promise((resolve, reject) => {
    cmd.on('error', reject).on('end', resolve).run();
  });
}

function getInfo(filePath) {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(filePath, (err, meta) => {
      if (err) return reject(err);
      const v = meta.streams.find(s => s.codec_type === 'video');
      const a = meta.streams.find(s => s.codec_type === 'audio');
      resolve({
        duration: parseFloat(meta.format.duration) || 0,
        width: v ? v.width : 720,
        height: v ? v.height : 1280,
        hasAudio: !!a,
      });
    });
  });
}

function tryDelete(p) { try { if (fs.existsSync(p)) fs.unlinkSync(p); } catch (_) {} }

// Single-pass: scale + pad to target size, re-encode at low quality
function scaleClip(input, output, w, h) {
  return run(
    ffmpeg(input)
      .output(output)
      .videoCodec('libx264')
      .audioCodec('aac')
      .outputOptions([
        '-threads', '1',
        '-preset', 'ultrafast',
        '-crf', '32',
        '-vf', `scale=${w}:${h}:force_original_aspect_ratio=decrease,pad=${w}:${h}:(ow-iw)/2:(oh-ih)/2,setsar=1`,
        '-ar', '44100',
        '-ac', '2',
        '-movflags', '+faststart',
        '-y',
      ])
  );
}

// Photo to 4-second video clip
function photoToClip(input, output, w, h) {
  return run(
    ffmpeg()
      .input(input)
      .inputOptions(['-loop', '1', '-t', '4'])
      .output(output)
      .videoCodec('libx264')
      .outputOptions([
        '-threads', '1',
        '-preset', 'ultrafast',
        '-crf', '32',
        '-vf', `scale=${w}:${h}:force_original_aspect_ratio=decrease,pad=${w}:${h}:(ow-iw)/2:(oh-ih)/2,setsar=1`,
        '-an',
        '-y',
      ])
  );
}

// Concat list of pre-encoded clips (stream copy - fast, low memory)
function concatClips(listFile, output) {
  return run(
    ffmpeg()
      .input(listFile)
      .inputOptions(['-f', 'concat', '-safe', '0'])
      .output(output)
      .outputOptions(['-c', 'copy', '-y'])
  );
}

// Add voiceover over existing video
function addVoiceover(videoPath, audioPath, output) {
  return run(
    ffmpeg()
      .input(videoPath)
      .input(audioPath)
      .output(output)
      .outputOptions([
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        '-threads', '1',
        '-y',
      ])
  );
}

async function generateCommentary(transcripts, style, openaiKey) {
  try {
    const client = new OpenAI({ apiKey: openaiKey });
    const text = transcripts.map((t, i) => `Clip ${i + 1}: "${t || '(no speech)'}"`).join('\n');
    const res = await client.chat.completions.create({
      model: 'gpt-4o-mini',
      max_tokens: 400,
      messages: [
        { role: 'system', content: `You create short social video commentary. Style: ${style}. Be concise. Return JSON only.` },
        { role: 'user', content: `${text}\n\nReturn: {"commentary":"30-60 word narration","title":"short title","hashtags":["tag1","tag2","tag3"],"description":"one sentence"}` },
      ],
    });
    return JSON.parse(res.choices[0].message.content.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim());
  } catch (e) {
    console.warn('Commentary failed:', e.message);
    return { commentary: '', title: 'My Video', hashtags: ['video'], description: '' };
  }
}

async function quickTranscribe(filePath, openaiKey) {
  try {
    const audioPath = filePath + '_tmp.mp3';
    await run(
      ffmpeg(filePath).output(audioPath)
        .outputOptions(['-threads', '1', '-ar', '16000', '-ac', '1', '-b:a', '32k', '-t', '60', '-y'])
    );
    if (!fs.existsSync(audioPath) || fs.statSync(audioPath).size < 1024) {
      tryDelete(audioPath);
      return '';
    }
    const client = new OpenAI({ apiKey: openaiKey });
    const t = await client.audio.transcriptions.create({
      file: fs.createReadStream(audioPath),
      model: 'whisper-1',
    });
    tryDelete(audioPath);
    return t.text || '';
  } catch (e) {
    console.warn('Transcribe failed:', e.message);
    return '';
  }
}

// ── main job processor ────────────────────────────────────────────────────────

async function processJob(jobId) {
  const job = jobs.get(jobId);
  if (!job) return;

  const jobDir = path.join(jobsDir, jobId);
  fs.mkdirSync(jobDir, { recursive: true });
  fs.mkdirSync(outputDir, { recursive: true });

  try {
    const openaiKey = job.openaiKey || OPENAI_API_KEY;
    if (!openaiKey) throw new Error('No OpenAI API key configured');

    const arConfigs = {
      '9:16': { w: 480, h: 854 },
      '16:9': { w: 854, h: 480 },
      '1:1':  { w: 480, h: 480 },
      '4:5':  { w: 480, h: 600 },
    };

    // Process one aspect ratio at a time to limit memory
    const outputUrls = {};
    const allTranscripts = [];

    for (let arIdx = 0; arIdx < job.aspectRatios.length; arIdx++) {
      const ar = job.aspectRatios[arIdx];
      const slug = ar.replace(':', '_');
      const cfg = arConfigs[ar] || { w: 480, h: 854 };

      job.progress = 5 + Math.round((arIdx / job.aspectRatios.length) * 70);
      job.message = `Processing ${ar} (${arIdx + 1}/${job.aspectRatios.length})...`;
      job.stage = 'rendering';
      job.status = 'processing';

      const scaledPaths = [];

      // Scale each clip to this aspect ratio
      for (let i = 0; i < job.fileIds.length; i++) {
        const fileId = job.fileIds[i];
        const srcPath = path.join(uploadsDir, fileId);
        if (!fs.existsSync(srcPath)) throw new Error(`Upload missing: ${fileId}`);

        const isImage = /\.(jpg|jpeg|png|heic|webp)$/i.test(fileId);
        const scaledPath = path.join(jobDir, `scaled_${arIdx}_${i}.mp4`);

        job.message = `Scaling clip ${i + 1}/${job.fileIds.length} for ${ar}...`;

        if (isImage) {
          await photoToClip(srcPath, scaledPath, cfg.w, cfg.h);
        } else {
          await scaleClip(srcPath, scaledPath, cfg.w, cfg.h);
        }

        scaledPaths.push(scaledPath);

        // Transcribe first clip of first aspect ratio only
        if (arIdx === 0 && allTranscripts.length <= i) {
          job.message = `Transcribing clip ${i + 1}...`;
          const text = await quickTranscribe(scaledPath, openaiKey);
          allTranscripts.push(text);
        }
      }

      // Concat all scaled clips
      job.message = `Concatenating clips for ${ar}...`;
      const listPath = path.join(jobDir, `list_${slug}.txt`);
      fs.writeFileSync(listPath, scaledPaths.map(p => `file '${p.replace(/\\/g, '/')}'`).join('\n'));

      const concatPath = path.join(jobDir, `concat_${slug}.mp4`);
      await concatClips(listPath, concatPath);
      scaledPaths.forEach(tryDelete);

      // Generate commentary once (after first aspect ratio transcription)
      if (arIdx === 0 && !job.plan) {
        job.message = 'Generating AI commentary...';
        job.stage = 'planning';
        job.plan = await generateCommentary(allTranscripts, job.style, openaiKey);

        // Generate TTS voiceover
        if (job.plan.commentary) {
          job.message = 'Generating voiceover...';
          job.stage = 'generating_tts';
          try {
            const client = new OpenAI({ apiKey: openaiKey });
            const res = await client.audio.speech.create({ model: 'tts-1', voice: 'nova', input: job.plan.commentary });
            const voPath = path.join(jobDir, 'vo.mp3');
            fs.writeFileSync(voPath, Buffer.from(await res.arrayBuffer()));
            job.voiceoverPath = voPath;
          } catch (e) { console.warn('TTS failed:', e.message); }
        }
      }

      // Add voiceover
      const finalPath = path.join(outputDir, `${jobId}_${slug}.mp4`);
      if (job.voiceoverPath && fs.existsSync(job.voiceoverPath)) {
        job.message = `Mixing audio for ${ar}...`;
        await addVoiceover(concatPath, job.voiceoverPath, finalPath);
        tryDelete(concatPath);
      } else {
        fs.renameSync(concatPath, finalPath);
      }

      outputUrls[ar] = `${BASE_URL}/jobs/${jobId}/download/${slug}`;
      console.log(`Completed ${ar}: ${finalPath}`);
    }

    // Clean up uploads
    job.fileIds.forEach(id => tryDelete(path.join(uploadsDir, id)));

    job.status = 'complete';
    job.progress = 100;
    job.stage = 'done';
    job.message = 'Done!';
    job.outputUrls = outputUrls;
    job.commentary = job.plan || { title: 'My Video', description: '', hashtags: [], commentary: '' };

    setTimeout(() => {
      try { fs.rmSync(jobDir, { recursive: true, force: true }); } catch (_) {}
      jobs.delete(jobId);
    }, 3600000);

  } catch (error) {
    console.error('Job error:', jobId, error.message);
    job.status = 'failed';
    job.error = error.message;
    job.message = error.message;
    job.fileIds.forEach(id => tryDelete(path.join(uploadsDir, id)));
  }
}

// ── routes ────────────────────────────────────────────────────────────────────

app.use(cors());
app.use(express.json());

app.get('/health', (req, res) => res.json({ status: 'ok', version: '2.0.0' }));

app.post('/upload', upload.array('files'), (req, res) => {
  if (!req.files?.length) return res.status(400).json({ error: 'No files' });
  // Rename to preserve extension
  const result = req.files.map(f => {
    const ext = path.extname(f.originalname).toLowerCase() || '.mp4';
    const newName = f.filename + ext;
    const newPath = path.join(uploadsDir, newName);
    fs.renameSync(f.path, newPath);
    return { id: newName, originalname: f.originalname };
  });
  res.json(result);
});

app.post('/jobs', (req, res) => {
  const { fileIds, style, aspectRatios, openaiKey } = req.body;
  if (!fileIds?.length) return res.status(400).json({ error: 'fileIds required' });
  if (!aspectRatios?.length) return res.status(400).json({ error: 'aspectRatios required' });
  const jobId = uuidv4();
  jobs.set(jobId, {
    id: jobId, fileIds, style: style || 'engaging', aspectRatios,
    openaiKey, status: 'queued', progress: 0, stage: 'queued', message: 'Queued...', outputUrls: {},
  });
  setImmediate(() => processJob(jobId));
  res.json({ jobId });
});

app.get('/jobs/:id', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: 'Not found' });
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