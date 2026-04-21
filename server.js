const express = require('express');
const multer = require('multer');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');
const { v4: uuidv4 } = require('uuid');
const OpenAI = require('openai');
const ffmpeg = require('fluent-ffmpeg');
const ffmpegStatic = require('ffmpeg-static');
const ffprobeStatic = require('ffprobe-static');

// Configure ffmpeg
ffmpeg.setFfmpegPath(ffmpegStatic);
ffmpeg.setFfprobePath(ffprobeStatic.path);

require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;
const BASE_URL = process.env.BASE_URL || `http://localhost:${PORT}`;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

// Ensure directories exist
const uploadsDir = path.join(__dirname, 'uploads');
const jobsDir = path.join(__dirname, 'jobs');
const outputDir = path.join(__dirname, 'outputs');

[uploadsDir, jobsDir, outputDir].forEach(dir => {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
});

// Storage setup
const storage = multer.diskStorage({
  destination: uploadsDir,
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}-${Math.random().toString(36).substring(7)}-${file.originalname}`;
    cb(null, uniqueName);
  },
});
const upload = multer({ storage });

// Job storage in memory
const jobs = new Map();

function pad(num, digits = 2) {
  return String(num).padStart(digits, '0');
}

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
      resolve({
        duration: metadata.format.duration || 0,
        width: stream.width || 1080,
        height: stream.height || 1920,
      });
    });
  });
}

function escapeFilterPath(filePath) {
  return filePath.replace(/\\/g, '/').replace(/:/g, '\\:');
}

async function transcribeClip(filePath, openaiKey) {
  const client = new OpenAI({ apiKey: openaiKey });
  try {
    const audioPath = filePath.replace(/\.[^.]+$/, '.wav');
    await new Promise((resolve, reject) => {
      ffmpeg(filePath).output(audioPath).on('error', reject).on('end', resolve).run();
    });
    const stats = fs.statSync(audioPath);
    if (stats.size < 2048) { fs.unlinkSync(audioPath); return { segments: [], text: '' }; }
    const audioFile = fs.createReadStream(audioPath);
    const transcript = await client.audio.transcriptions.create({
      file: audioFile, model: 'whisper-1', response_format: 'verbose_json', timestamp_granularity: 'segment',
    });
    fs.unlinkSync(audioPath);
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
    const ffmpegBin = ffmpegStatic;
    const child = spawn(ffmpegBin, ['-i', filePath, '-af', 'silencedetect=noise=-35dB:d=1.5', '-f', 'null', '-'], { stdio: ['ignore', 'pipe', 'pipe'] });
    let stderr = '';
    child.stderr.on('data', (data) => { stderr += data.toString(); });
    child.on('close', () => {
      const lines = stderr.split('\n');
      for (const line of lines) {
        const startMatch = line.match(/silence_start:\s*([\d.]+)/);
        const endMatch = line.match(/silence_end:\s*([\d.]+)/);
        if (startMatch) currentStart = parseFloat(startMatch[1]);
        if (endMatch && currentStart !== null) {
          silenceRanges.push({ start: currentStart, end: parseFloat(endMatch[1]) });
          currentStart = null;
        }
      }
      resolve(silenceRanges);
    });
    child.on('error', () => resolve([]));
  });
}

function invertSilenceRanges(silenceRanges, totalDuration) {
  const keepRanges = [];
  let currentPos = 0;
  for (const silence of silenceRanges) {
    const paddedStart = Math.max(0, silence.start - 0.3);
    const paddedEnd = Math.min(totalDuration, silence.end + 0.3);
    if (currentPos < paddedStart) keepRanges.push({ start: currentPos, end: paddedStart });
    currentPos = paddedEnd;
  }
  if (currentPos < totalDuration) keepRanges.push({ start: currentPos, end: totalDuration });
  return keepRanges;
}

async function normalizeClip(filePath, outputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(filePath).output(outputPath).videoCodec('libx264').audioCodec('aac').on('error', reject).on('end', resolve).run();
  });
}

async function photoToVideo(filePath, outputPath, duration = 5) {
  return new Promise((resolve, reject) => {
    ffmpeg().input(filePath).inputOptions(['-loop', '1', '-t', String(duration)])
      .output(outputPath).videoCodec('libx264').size('1080x1920')
      .videoFilters('scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2')
      .on('error', reject).on('end', resolve).run();
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
      const fileId = job.fileIds[i];
      const filePath = path.join(uploadsDir, fileId);
      if (!fs.existsSync(filePath)) throw new Error(`File not found: ${fileId}`);
      job.progress = 5 + Math.round((i / job.fileIds.length) * 15);
      job.message = `Normalizing clip ${i + 1} of ${job.fileIds.length}...`;
      const isVideo = ['.mp4', '.mov', '.avi', '.mkv'].some(ext => fileId.toLowerCase().endsWith(ext));
      const isImage = ['.jpg', '.jpeg', '.png', '.heic', '.webp'].some(ext => fileId.toLowerCase().endsWith(ext));
      let normalizedPath;
      if (isVideo) {
        normalizedPath = path.join(jobDir, `normalized_${i}.mp4`);
        await normalizeClip(filePath, normalizedPath);
      } else if (isImage) {
        normalizedPath = path.join(jobDir, `normalized_${i}.mp4`);
        await photoToVideo(filePath, normalizedPath, 5);
      } else {
        throw new Error(`Unsupported file type: ${fileId}`);
      }
      const info = await getVideoInfo(normalizedPath);
      normalizedClips.push({ index: i, path: normalizedPath, duration: info.duration, width: info.width, height: info.height });
    }

    // Stage 2: Transcribe
    job.stage = 'transcribing';
    const clipTranscripts = [];
    for (let i = 0; i < normalizedClips.length; i++) {
      job.progress = 20 + Math.round((i / normalizedClips.length) * 15);
      job.message = `Transcribing clip ${i + 1} of ${normalizedClips.length}...`;
      const transcript = await transcribeClip(normalizedClips[i].path, openaiKey);
      clipTranscripts.push({ index: i, text: transcript.text, segments: transcript.segments });
    }

    // Stage 3: Analyze silence
    job.stage = 'analyzing';
    const clipKeepRanges = [];
    for (let i = 0; i < normalizedClips.length; i++) {
      job.progress = 35 + Math.round((i / normalizedClips.length) * 15);
      job.message = `Analyzing clip ${i + 1} of ${normalizedClips.length}...`;
      const silenceRanges = await detectSilence(normalizedClips[i].path);
      const keepRanges = invertSilenceRanges(silenceRanges, normalizedClips[i].duration);
      clipKeepRanges.push({ index: i, keepRanges: keepRanges.length > 0 ? keepRanges : [{ start: 0, end: normalizedClips[i].duration }] });
    }

    // Stage 4: Plan editing
    job.stage = 'planning'; job.progress = 50; job.message = 'Planning edits with AI...';
    let editPlan = { editOrder: normalizedClips.map((_, i) => i), commentary: '', title: 'ClipCraft Video', hashtags: ['clipcraft'], description: 'Created with ClipCraft' };
    try {
      const client = new OpenAI({ apiKey: openaiKey });
      const summaryPrompt = clipTranscripts.map((t, i) => `Clip ${i}: "${t.text || '(no audio)'}"`).join('\n');
      const response = await client.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [
          { role: 'system', content: `You are a video editing AI. Style: ${job.style || 'engaging'}. Return JSON only, no markdown.` },
          { role: 'user', content: `${summaryPrompt}\n\nReturn JSON: {"editOrder":[0,1,2...],"commentary":"narration text","title":"video title","hashtags":["tag1","tag2"],"description":"description text"}` },
        ],
      });
      const content = response.choices[0].message.content.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
      editPlan = JSON.parse(content);
    } catch (error) { console.warn('AI planning failed, using default:', error.message); }

    // Stage 5: TTS
    job.stage = 'generating_tts'; job.progress = 60; job.message = 'Generating voiceover...';
    let voiceoverPath = null;
    if (editPlan.commentary) {
      try {
        const client = new OpenAI({ apiKey: openaiKey });
        const mp3Path = path.join(jobDir, 'voiceover.mp3');
        const response = await client.audio.speech.create({ model: 'tts-1', voice: 'nova', input: editPlan.commentary });
        const buffer = await response.arrayBuffer();
        fs.writeFileSync(mp3Path, Buffer.from(buffer));
        voiceoverPath = mp3Path;
      } catch (error) { console.warn('TTS failed:', error.message); }
    }

    // Stage 6: Render
    job.stage = 'rendering';
    const outputUrls = {};
    const aspectRatioConfigs = { '9:16': { width: 1080, height: 1920 }, '16:9': { width: 1920, height: 1080 }, '1:1': { width: 1080, height: 1080 }, '4:5': { width: 1080, height: 1350 } };

    for (let ai = 0; ai < job.aspectRatios.length; ai++) {
      const aspectRatio = job.aspectRatios[ai];
      const config = aspectRatioConfigs[aspectRatio];
      job.progress = 70 + Math.round((ai / job.aspectRatios.length) * 25);
      job.message = `Rendering ${aspectRatio}...`;

      const concatPath = path.join(jobDir, `concat_${aspectRatio.replace(':','_')}.txt`);
      let concatContent = '';

      for (const clipIdx of editPlan.editOrder) {
        if (clipIdx >= normalizedClips.length) continue;
        const clipInfo = normalizedClips[clipIdx];
        const keepRanges = clipKeepRanges[clipIdx].keepRanges;
        if (keepRanges.length === 0) continue;

        if (keepRanges.length === 1) {
          const tempPath = path.join(jobDir, `trimmed_${clipIdx}_${aspectRatio.replace(':','_')}.mp4`);
          await new Promise((resolve, reject) => {
            ffmpeg(clipInfo.path).seekInput(keepRanges[0].start).duration(keepRanges[0].end - keepRanges[0].start)
              .output(tempPath).videoCodec('libx264').audioCodec('aac').on('error', reject).on('end', resolve).run();
          });
          concatContent += `file '${tempPath.replace(/\\/g, '/')}'\n`;
        } else {
          const segmentPaths = [];
          for (let si = 0; si < keepRanges.length; si++) {
            const range = keepRanges[si];
            const segPath = path.join(jobDir, `segment_${clipIdx}_${si}_${aspectRatio.replace(':','_')}.mp4`);
            segmentPaths.push(segPath);
            await new Promise((resolve, reject) => {
              ffmpeg(clipInfo.path).seekInput(range.start).duration(range.end - range.start)
                .output(segPath).videoCodec('libx264').audioCodec('aac').on('error', reject).on('end', resolve).run();
            });
          }
          const segConcatPath = path.join(jobDir, `segconcat_${clipIdx}_${aspectRatio.replace(':','_')}.txt`);
          fs.writeFileSync(segConcatPath, segmentPaths.map(p => `file '${p.replace(/\\/g, '/')}'`).join('\n'));
          const mergedPath = path.join(jobDir, `merged_${clipIdx}_${aspectRatio.replace(':','_')}.mp4`);
          await new Promise((resolve, reject) => {
            ffmpeg().input(segConcatPath).inputOptions(['-f', 'concat', '-safe', '0'])
              .output(mergedPath).videoCodec('libx264').audioCodec('aac').on('error', reject).on('end', resolve).run();
          });
          concatContent += `file '${mergedPath.replace(/\\/g, '/')}'\n`;
        }
      }

      fs.writeFileSync(concatPath, concatContent);

      const concatedPath = path.join(jobDir, `concatenated_${aspectRatio.replace(':','_')}.mp4`);
      await new Promise((resolve, reject) => {
        ffmpeg().input(concatPath).inputOptions(['-f', 'concat', '-safe', '0'])
          .output(concatedPath).videoCodec('libx264').audioCodec('aac').on('error', reject).on('end', resolve).run();
      });

      const scaledPath = path.join(jobDir, `scaled_${aspectRatio.replace(':','_')}.mp4`);
      await new Promise((resolve, reject) => {
        ffmpeg(concatedPath).output(scaledPath)
          .videoFilters(`scale=${config.width}:${config.height}:force_original_aspect_ratio=decrease,pad=${config.width}:${config.height}:(ow-iw)/2:(oh-ih)/2`)
          .videoCodec('libx264').audioCodec('aac').on('error', reject).on('end', resolve).run();
      });

      let subtitledPath = scaledPath;
      if (editPlan.commentary && clipTranscripts.some(t => t.segments && t.segments.length > 0)) {
        const srtPath = path.join(jobDir, `subs_${aspectRatio.replace(':','_')}.srt`);
        let srtContent = '';
        let srtIndex = 1;
        for (const transcript of clipTranscripts) {
          for (const segment of (transcript.segments || [])) {
            srtContent += `${srtIndex}\n${formatSRTTime(segment.start)} --> ${formatSRTTime(segment.end)}\n${segment.text.trim()}\n\n`;
            srtIndex++;
          }
        }
        if (srtContent) {
          fs.writeFileSync(srtPath, srtContent);
          subtitledPath = path.join(jobDir, `subtitled_${aspectRatio.replace(':','_')}.mp4`);
          await new Promise((resolve, reject) => {
            const escapedSrt = srtPath.replace(/\\/g, '/').replace(/:/g, '\\:');
            ffmpeg(scaledPath).output(subtitledPath)
              .videoFilters(`subtitles='${escapedSrt}'`)
              .videoCodec('libx264').audioCodec('aac').on('error', (err) => { console.warn('Subtitle burn failed:', err.message); resolve(); subtitledPath = scaledPath; }).on('end', resolve).run();
          });
        }
      }

      const finalPath = path.join(outputDir, `${jobId}_${aspectRatio.replace(':','_')}.mp4`);
      if (voiceoverPath) {
        await new Promise((resolve, reject) => {
          ffmpeg().input(subtitledPath).input(voiceoverPath)
            .complexFilter('[0:a][1:a]amix=inputs=2:duration=first[a]')
            .map('0:v').map('[a]')
            .output(finalPath).videoCodec('copy').audioCodec('aac')
            .on('error', reject).on('end', resolve).run();
        });
      } else {
        fs.copyFileSync(subtitledPath, finalPath);
      }

      outputUrls[aspectRatio] = `${BASE_URL}/jobs/${jobId}/download/${aspectRatio.replace(':', '_')}`;
    }

    job.status = 'complete'; job.progress = 100; job.stage = 'done'; job.message = 'Export complete!';
    job.outputUrls = outputUrls;
    job.commentary = { title: editPlan.title, description: editPlan.description, hashtags: editPlan.hashtags, commentary: editPlan.commentary };

    setTimeout(() => {
      try { fs.rmSync(path.join(jobsDir, jobId), { recursive: true, force: true }); jobs.delete(jobId); } catch (e) {}
    }, 3600000);
  } catch (error) {
    job.status = 'failed'; job.error = error.message; job.message = error.message;
    console.error('Job failed:', error);
  }
}

app.use(cors());
app.use(express.json());

app.get('/health', (req, res) => res.json({ status: 'ok', version: '1.0.0' }));

app.post('/upload', upload.array('files'), (req, res) => {
  if (!req.files || req.files.length === 0) return res.status(400).json({ error: 'No files uploaded' });
  res.json(req.files.map(file => ({ id: file.filename, originalname: file.originalname, mimetype: file.mimetype })));
});

app.post('/jobs', (req, res) => {
  const { fileIds, style, aspectRatios, maxDuration, openaiKey } = req.body;
  if (!fileIds || !Array.isArray(fileIds) || fileIds.length === 0) return res.status(400).json({ error: 'fileIds required' });
  if (!aspectRatios || !Array.isArray(aspectRatios) || aspectRatios.length === 0) return res.status(400).json({ error: 'aspectRatios required' });
  const jobId = uuidv4();
  jobs.set(jobId, { id: jobId, fileIds, style: style || 'engaging', aspectRatios, maxDuration: maxDuration || 0, openaiKey, status: 'queued', progress: 0, stage: 'queued', message: 'Job queued...', outputUrls: {} });
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
  if (!fs.existsSync(filePath)) return res.status(404).json({ error: 'Output file not found' });
  res.download(filePath, `clipcraft_${aspect}.mp4`);
});

app.listen(PORT, () => {
  console.log(`ClipCraft backend running on port ${PORT}`);
});