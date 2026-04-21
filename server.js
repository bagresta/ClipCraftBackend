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

function ensureDirs() {
  [uploadsDir, jobsDir, outputDir].forEach(dir => {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  });
}
ensureDirs();

const storage = multer.diskStorage({
  destination: (req, file, cb) => { ensureDirs(); cb(null, uploadsDir); },
  filename: (req, file, cb) => cb(null, `${Date.now()}-${Math.random().toString(36).substring(7)}-${file.originalname}`),
});
const upload = multer({ storage, limits: { fileSize: 500 * 1024 * 1024 } });
const jobs = new Map();

function pad(n, d=2) { return String(n).padStart(d,'0'); }
function formatSRTTime(s) {
  return `${pad(Math.floor(s/3600))}:${pad(Math.floor((s%3600)/60))}:${pad(Math.floor(s%60))},${pad(Math.round((s%1)*1000),3)}`;
}

function getVideoInfo(filePath) {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(filePath, (err, meta) => {
      if (err) return reject(err);
      const v = meta.streams.find(s => s.codec_type === 'video') || meta.streams[0];
      const a = meta.streams.find(s => s.codec_type === 'audio');
      resolve({ duration: meta.format.duration || 0, width: v?.width || 720, height: v?.height || 1280, hasAudio: !!a });
    });
  });
}

const BASE_OPTS = ['-threads', '1', '-preset', 'ultrafast', '-crf', '28', '-y'];

function runFfmpeg(cmd) {
  return new Promise((resolve, reject) => {
    cmd.outputOptions(['-y']).on('error', reject).on('end', resolve).run();
  });
}

async function normalizeClip(input, output) {
  ensureDirs();
  await runFfmpeg(
    ffmpeg(input).output(output)
      .videoCodec('libx264').audioCodec('aac')
      .outputOptions([...BASE_OPTS, '-vf', 'scale=720:-2', '-movflags', '+faststart'])
  );
}

async function photoToVideo(input, output) {
  ensureDirs();
  await runFfmpeg(
    ffmpeg().input(input).inputOptions(['-loop','1','-t','5'])
      .output(output).videoCodec('libx264').audioCodec('aac')
      .outputOptions([...BASE_OPTS, '-vf', 'scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2', '-movflags', '+faststart'])
  );
}

async function transcribeClip(filePath, openaiKey) {
  const client = new OpenAI({ apiKey: openaiKey });
  try {
    const audioPath = filePath.replace(/\.[^.]+$/, '_audio.mp3');
    await runFfmpeg(ffmpeg(filePath).output(audioPath).outputOptions(['-threads','1','-q:a','5','-map','a','-y']));
    if (!fs.existsSync(audioPath) || fs.statSync(audioPath).size < 2048) {
      try { fs.unlinkSync(audioPath); } catch(_){}
      return { segments: [], text: '' };
    }
    const t = await client.audio.transcriptions.create({ file: fs.createReadStream(audioPath), model: 'whisper-1', response_format: 'verbose_json', timestamp_granularity: 'segment' });
    try { fs.unlinkSync(audioPath); } catch(_){}
    return { segments: t.segments || [], text: t.text || '' };
  } catch(e) { console.error('Transcribe error:', e.message); return { segments: [], text: '' }; }
}

function detectSilence(filePath) {
  return new Promise(resolve => {
    const ranges = []; let start = null;
    const child = spawn(ffmpegStatic, ['-i', filePath, '-af', 'silencedetect=noise=-35dB:d=1.5', '-f', 'null', '-'], { stdio: ['ignore','pipe','pipe'] });
    let stderr = '';
    child.stderr.on('data', d => stderr += d.toString());
    child.on('close', () => {
      for (const line of stderr.split('\n')) {
        const s = line.match(/silence_start:\s*([\d.]+)/);
        const e = line.match(/silence_end:\s*([\d.]+)/);
        if (s) start = parseFloat(s[1]);
        if (e && start !== null) { ranges.push({ start, end: parseFloat(e[1]) }); start = null; }
      }
      resolve(ranges);
    });
    child.on('error', () => resolve([]));
  });
}

function invertSilenceRanges(ranges, duration) {
  const keep = []; let pos = 0;
  for (const s of ranges) {
    const ps = Math.max(0, s.start - 0.3), pe = Math.min(duration, s.end + 0.3);
    if (pos < ps) keep.push({ start: pos, end: ps });
    pos = pe;
  }
  if (pos < duration) keep.push({ start: pos, end: duration });
  return keep;
}

async function trimClip(input, output, start, duration) {
  await runFfmpeg(
    ffmpeg(input).seekInput(start).duration(duration).output(output)
      .videoCodec('libx264').audioCodec('aac').outputOptions(BASE_OPTS)
  );
}

async function concatClips(listFile, output) {
  await runFfmpeg(
    ffmpeg().input(listFile).inputOptions(['-f','concat','-safe','0'])
      .output(output).videoCodec('libx264').audioCodec('aac').outputOptions(BASE_OPTS)
  );
}

async function scaleAndPad(input, output, w, h) {
  await runFfmpeg(
    ffmpeg(input).output(output)
      .videoCodec('libx264').audioCodec('aac')
      .outputOptions(BASE_OPTS)
      .videoFilters(`scale=${w}:${h}:force_original_aspect_ratio=decrease,pad=${w}:${h}:(ow-iw)/2:(oh-ih)/2`)
  );
}

async function mixVoiceover(videoPath, voiceoverPath, output, hasAudio) {
  ensureDirs();
  if (hasAudio) {
    await runFfmpeg(
      ffmpeg().input(videoPath).input(voiceoverPath)
        .complexFilter('[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[a]')
        .map('0:v').map('[a]')
        .output(output).videoCodec('copy').audioCodec('aac')
        .outputOptions(['-threads','1','-y'])
    );
  } else {
    await runFfmpeg(
      ffmpeg().input(videoPath).input(voiceoverPath)
        .map('0:v').map('1:a')
        .output(output).videoCodec('copy').audioCodec('aac')
        .outputOptions(['-threads','1','-shortest','-y'])
    );
  }
}

async function processJob(jobId) {
  const job = jobs.get(jobId);
  if (!job) return;
  try {
    ensureDirs();
    const jobDir = path.join(jobsDir, jobId);
    if (!fs.existsSync(jobDir)) fs.mkdirSync(jobDir, { recursive: true });
    const openaiKey = job.openaiKey || OPENAI_API_KEY;
    if (!openaiKey) throw new Error('No OpenAI API key provided');

    job.stage = 'normalizing'; job.status = 'processing';
    const normalizedClips = [];
    for (let i = 0; i < job.fileIds.length; i++) {
      job.progress = 5 + Math.round((i / job.fileIds.length) * 15);
      job.message = `Normalizing clip ${i+1} of ${job.fileIds.length}...`;
      const filePath = path.join(uploadsDir, job.fileIds[i]);
      if (!fs.existsSync(filePath)) throw new Error(`File not found: ${job.fileIds[i]}`);
      const isImage = ['.jpg','.jpeg','.png','.heic','.webp'].some(e => job.fileIds[i].toLowerCase().endsWith(e));
      const normPath = path.join(jobDir, `norm_${i}.mp4`);
      if (isImage) await photoToVideo(filePath, normPath);
      else await normalizeClip(filePath, normPath);
      try { fs.unlinkSync(filePath); } catch(_){}
      const info = await getVideoInfo(normPath);
      normalizedClips.push({ index: i, path: normPath, duration: info.duration, hasAudio: info.hasAudio });
    }

    job.stage = 'transcribing';
    const clipTranscripts = [];
    for (let i = 0; i < normalizedClips.length; i++) {
      job.progress = 20 + Math.round((i / normalizedClips.length) * 15);
      job.message = `Transcribing clip ${i+1}...`;
      const t = await transcribeClip(normalizedClips[i].path, openaiKey);
      clipTranscripts.push({ index: i, text: t.text, segments: t.segments });
    }

    job.stage = 'analyzing';
    const clipKeepRanges = [];
    for (let i = 0; i < normalizedClips.length; i++) {
      job.progress = 35 + Math.round((i / normalizedClips.length) * 15);
      job.message = `Analyzing clip ${i+1}...`;
      const silence = await detectSilence(normalizedClips[i].path);
      const keep = invertSilenceRanges(silence, normalizedClips[i].duration);
      clipKeepRanges.push({ keepRanges: keep.length > 0 ? keep : [{ start: 0, end: normalizedClips[i].duration }] });
    }

    job.stage = 'planning'; job.progress = 50; job.message = 'AI planning...';
    let plan = { editOrder: normalizedClips.map((_,i)=>i), commentary: '', title: 'ClipCraft Video', hashtags: ['clipcraft'], description: 'Created with ClipCraft' };
    try {
      const client = new OpenAI({ apiKey: openaiKey });
      const prompt = clipTranscripts.map((t,i)=>`Clip ${i}: "${t.text||'(no speech)'}"` ).join('\n');
      const res = await client.chat.completions.create({
        model: 'gpt-4o-mini', max_tokens: 500,
        messages: [
          { role: 'system', content: `Video editing AI. Style: ${job.style||'engaging'}. Return JSON only.` },
          { role: 'user', content: `${prompt}\n\nReturn: {"editOrder":[0,1],"commentary":"narration","title":"title","hashtags":["tag"],"description":"desc"}` },
        ],
      });
      plan = JSON.parse(res.choices[0].message.content.replace(/```json\n?/g,'').replace(/```\n?/g,'').trim());
    } catch(e) { console.warn('AI plan failed:', e.message); }

    job.stage = 'generating_tts'; job.progress = 60; job.message = 'Generating voiceover...';
    let voiceoverPath = null;
    if (plan.commentary) {
      try {
        const client = new OpenAI({ apiKey: openaiKey });
        const mp3 = path.join(jobDir, 'voiceover.mp3');
        const res = await client.audio.speech.create({ model: 'tts-1', voice: 'nova', input: plan.commentary.substring(0, 4096) });
        fs.writeFileSync(mp3, Buffer.from(await res.arrayBuffer()));
        voiceoverPath = mp3;
      } catch(e) { console.warn('TTS failed:', e.message); }
    }

    job.stage = 'rendering';
    const arConfigs = { '9:16':{w:720,h:1280}, '16:9':{w:1280,h:720}, '1:1':{w:720,h:720}, '4:5':{w:720,h:900} };
    const outputUrls = {};

    for (let ai = 0; ai < job.aspectRatios.length; ai++) {
      const ar = job.aspectRatios[ai];
      const slug = ar.replace(':','_');
      const cfg = arConfigs[ar] || { w:720, h:1280 };
      job.progress = 65 + Math.round((ai / job.aspectRatios.length) * 28);
      job.message = `Rendering ${ar}...`;

      const concatLines = [];
      for (const idx of (plan.editOrder || normalizedClips.map((_,i)=>i))) {
        if (idx >= normalizedClips.length) continue;
        const clip = normalizedClips[idx];
        const keepRanges = clipKeepRanges[idx]?.keepRanges || [{ start:0, end:clip.duration }];
        for (let ri = 0; ri < keepRanges.length; ri++) {
          const { start, end } = keepRanges[ri];
          const dur = end - start;
          if (dur < 0.5) continue;
          const trimPath = path.join(jobDir, `trim_${idx}_${ri}_${slug}.mp4`);
          await trimClip(clip.path, trimPath, start, dur);
          concatLines.push(`file '${trimPath.replace(/\\/g,'/')}'`);
        }
      }

      if (concatLines.length === 0) throw new Error('No segments after trimming');
      const concatFile = path.join(jobDir, `list_${slug}.txt`);
      fs.writeFileSync(concatFile, concatLines.join('\n'));

      const concatOut = path.join(jobDir, `concat_${slug}.mp4`);
      await concatClips(concatFile, concatOut);

      const scaledPath = path.join(jobDir, `scaled_${slug}.mp4`);
      await scaleAndPad(concatOut, scaledPath, cfg.w, cfg.h);
      try { fs.unlinkSync(concatOut); } catch(_){}

      ensureDirs(); // ensure output dir exists before writing final file
      const finalPath = path.join(outputDir, `${jobId}_${slug}.mp4`);
      console.log('Writing final to:', finalPath);

      if (voiceoverPath) {
        const scaledInfo = await getVideoInfo(scaledPath);
        await mixVoiceover(scaledPath, voiceoverPath, finalPath, scaledInfo.hasAudio);
      } else {
        fs.copyFileSync(scaledPath, finalPath);
      }
      try { fs.unlinkSync(scaledPath); } catch(_){}

      outputUrls[ar] = `${BASE_URL}/jobs/${jobId}/download/${slug}`;
    }

    for (const c of normalizedClips) { try { fs.unlinkSync(c.path); } catch(_){} }

    job.status = 'complete'; job.progress = 100; job.stage = 'done'; job.message = 'Done!';
    job.outputUrls = outputUrls;
    job.commentary = { title: plan.title, description: plan.description, hashtags: plan.hashtags, commentary: plan.commentary };

    setTimeout(() => { try { fs.rmSync(path.join(jobsDir,jobId),{recursive:true,force:true}); jobs.delete(jobId); } catch(_){} }, 3600000);
  } catch(error) {
    console.error('Job failed:', error);
    job.status = 'failed'; job.error = error.message; job.message = error.message;
  }
}

app.use(cors());
app.use(express.json());
app.get('/health', (req,res) => res.json({ status:'ok', version:'1.0.0' }));

app.post('/upload', upload.array('files'), (req,res) => {
  if (!req.files?.length) return res.status(400).json({ error:'No files' });
  res.json(req.files.map(f => ({ id:f.filename, originalname:f.originalname, mimetype:f.mimetype })));
});

app.post('/jobs', (req,res) => {
  const { fileIds, style, aspectRatios, maxDuration, openaiKey } = req.body;
  if (!fileIds?.length) return res.status(400).json({ error:'fileIds required' });
  if (!aspectRatios?.length) return res.status(400).json({ error:'aspectRatios required' });
  const jobId = uuidv4();
  jobs.set(jobId, { id:jobId, fileIds, style:style||'engaging', aspectRatios, maxDuration:maxDuration||0, openaiKey, status:'queued', progress:0, stage:'queued', message:'Queued...', outputUrls:{} });
  setImmediate(() => processJob(jobId));
  res.json({ jobId });
});

app.get('/jobs/:id', (req,res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error:'Not found' });
  res.json({ id:job.id, status:job.status, progress:job.progress, stage:job.stage, message:job.message, outputUrls:job.outputUrls, commentary:job.commentary, error:job.error });
});

app.get('/jobs/:jobId/download/:aspect', (req,res) => {
  const { jobId, aspect } = req.params;
  if (!jobs.get(jobId)) return res.status(404).json({ error:'Job not found' });
  const filePath = path.join(outputDir, `${jobId}_${aspect}.mp4`);
  if (!fs.existsSync(filePath)) return res.status(404).json({ error:'File not found' });
  res.download(filePath, `clipcraft_${aspect}.mp4`);
});

app.listen(PORT, () => console.log(`ClipCraft backend on port ${PORT}`));