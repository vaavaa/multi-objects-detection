import { useState, useCallback, useEffect, useRef } from 'react';

const API_BASE = import.meta.env.VITE_HEREYOUARE_URL || 'http://localhost:8082';
const BLINK_CYCLE_MS = 2000;
const BLINK_MIN_ALPHA = 0.3;
const V2_FADE_OUT_MS = 500;  // старые рамки (v1): 100% → 20% за 0.5 с
const V2_FADE_IN_MS = 500;   // новые рамки (v2): 30% → 100% за 0.5 с
const V2_FADE_OUT_MIN = 0.2;
const V2_FADE_IN_START = 0.3;

const DEBUG_DRAW = false; // подробные логи отрисовки и опроса (отключить в проде)

function log(...args) {
  if (DEBUG_DRAW) console.log('[HereYouAre]', ...args);
}

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('idle');
  const [detections, setDetections] = useState([]);
  const [resultImageSize, setResultImageSize] = useState(null);
  const [resultVersion, setResultVersion] = useState(null); // null | 'v1' | 'v2' — что сейчас отображаем; только v2 завершает задачу
  const [error, setError] = useState(null);
  const pollIntervalMs = 250;

  const v2PayloadRef = useRef(null);
  const [v2Received, setV2Received] = useState(0);
  const v1AnimationStartedRef = useRef(false);

  const onFile = useCallback((e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setJobId(null);
    setDetections([]);
    setResultImageSize(null);
    setResultVersion(null);
    v1AnimationStartedRef.current = false;
    setStatus('idle');
    setError(null);
  }, []);

  const startJob = useCallback(async () => {
    if (!file) return;
    setStatus('uploading');
    try {
      const form = new FormData();
      form.append('file', file);
      const r = await fetch(`${API_BASE}/v1/detect/async`, {
        method: 'POST',
        body: form,
      });
      if (!r.ok) throw new Error(await r.text());
      const { job_id } = await r.json();
      setJobId(job_id);
      setDetections([]);
      setResultImageSize(null);
      setResultVersion(null);
      v2PayloadRef.current = null;
      v1AnimationStartedRef.current = false;
      setStatus('processing');
    } catch (err) {
      setError(String(err));
      setStatus('idle');
    }
  }, [file]);

  // Опрос не прекращаем после v1 — ждём v2 (status === 'done'); только v2 завершает задачу
  useEffect(() => {
    if (!jobId || status !== 'processing') return;
    log('poll: start', { jobId });
    const t = setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/v1/job/${jobId}`);
        if (!r.ok) return;
        const data = await r.json();
        if (data.status === 'done') {
          const v2Detections = data.detections || [];
          const v2Image =
            data.image && typeof data.image.width === 'number' && typeof data.image.height === 'number'
              ? { w: data.image.width, h: data.image.height }
              : null;
          log('poll: v2 received', { detectionsCount: v2Detections.length, image: v2Image });
          v2PayloadRef.current = { detections: v2Detections, image: v2Image };
          setV2Received((k) => k + 1);
          return;
        }
        if (data.status === 'failed') {
          log('poll: failed', data.error);
          setError(data.error || 'Job failed');
          setStatus('idle');
          return;
        }
        if (data.status === 'processing' && (data.version === 'v1' || (data.detections && data.detections.length > 0))) {
          const list = data.detections || [];
          if (list.length > 0) {
            log('poll: v1 received', { detectionsCount: list.length, version: data.version });
            setDetections(list);
            setResultImageSize(
              data.image && typeof data.image.width === 'number' && typeof data.image.height === 'number'
                ? { w: data.image.width, h: data.image.height }
                : null
            );
            setResultVersion('v1');
          }
        }
      } catch (e) {
        log('poll: error', e);
        setError(String(e));
        setStatus('idle');
      }
    }, pollIntervalMs);
    return () => {
      log('poll: stop');
      clearInterval(t);
    };
  }, [jobId, status]);

  const drawCorner = (ctx, x1, y1, x2, y2, color, lineWidth, opacity = 1) => {
    const w = x2 - x1;
    const h = y2 - y1;
    const segW = Math.max(1, Math.floor(w * 0.15));
    const segH = Math.max(1, Math.floor(h * 0.15));
    ctx.save();
    if (opacity < 1) ctx.globalAlpha = opacity;
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 + segW, y1);
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1, y1 + segH);
    ctx.moveTo(x2 - segW, y1);
    ctx.lineTo(x2, y1);
    ctx.moveTo(x2, y1);
    ctx.lineTo(x2, y1 + segH);
    ctx.moveTo(x2, y2 - segH);
    ctx.lineTo(x2, y2);
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - segW, y2);
    ctx.moveTo(x1, y2 - segH);
    ctx.lineTo(x1, y2);
    ctx.moveTo(x1, y2);
    ctx.lineTo(x1 + segW, y2);
    ctx.stroke();
    ctx.restore();
  };

  const [imageDim, setImageDim] = useState({ w: 0, h: 0 });
  const canvasRef = useRef(null);
  const renderRef = useRef({ img: null, ctx: null, w: 0, h: 0, dpr: 1 });
  const blinkRef = useRef({ running: false, startAt: 0, rafId: 0 });
  const doneTimerRef = useRef(0);
  const v2TransitionRef = useRef(null); // { phase: 'fadeOut'|'fadeIn', oldDetections, newDetections, imageSize, startTime, rafId }
  const [stripHeightPx, setStripHeightPx] = useState(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      const h = canvas.getBoundingClientRect().height;
      setStripHeightPx(Math.max(56, Math.round(h * 0.25)));
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [preview]);

  useEffect(() => {
    // Подготовка canvas + хранение img/ctx для дальнейшей анимации
    if (!preview || !canvasRef.current) return;

    if (doneTimerRef.current) window.clearTimeout(doneTimerRef.current);
    if (blinkRef.current.rafId) cancelAnimationFrame(blinkRef.current.rafId);
    if (v2TransitionRef.current?.rafId) cancelAnimationFrame(v2TransitionRef.current.rafId);
    blinkRef.current = { running: false, startAt: 0, rafId: 0 };
    v2TransitionRef.current = null;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const imgW = img.naturalWidth;
      const imgH = img.naturalHeight;
      const dpr = window.devicePixelRatio || 1;

      canvas.width = imgW * dpr;
      canvas.height = imgH * dpr;

      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, imgW, imgH);
      ctx.drawImage(img, 0, 0);

      setImageDim({ w: imgW, h: imgH });
      renderRef.current = { img, ctx, w: imgW, h: imgH, dpr };
    };
    img.src = preview;
  }, [preview]);

  useEffect(() => {
    const { img, ctx, w: imgW, h: imgH } = renderRef.current;
    if (!img || !ctx) return;

    log('draw: effect run', { status, resultVersion, detectionsCount: detections?.length, hasPayload: !!v2PayloadRef.current, hasTrans: !!v2TransitionRef.current });

    const apiW = resultImageSize?.w > 0 ? resultImageSize.w : imgW;
    const apiH = resultImageSize?.h > 0 ? resultImageSize.h : imgH;
    const scaleX = imgW / apiW;
    const scaleY = imgH / apiH;

    const baseLineW = Math.max(2, Math.min(imgW, imgH) / 400);
    const lineW = baseLineW * 1.5;

    const drawBase = () => {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, imgW, imgH);
      ctx.drawImage(img, 0, 0);
    };

    const drawInitialCornerBox = (alpha) => {
      const cx = imgW / 2;
      const cy = imgH / 2;
      const side = Math.min(imgW, imgH) * 0.3;
      const x1 = Math.round(cx - side / 2);
      const y1 = Math.round(cy - side / 2);
      const x2 = Math.round(cx + side / 2);
      const y2 = Math.round(cy + side / 2);
      const a = Math.max(BLINK_MIN_ALPHA, Math.min(1, alpha));
      drawCorner(ctx, x1, y1, x2, y2, `rgba(100, 180, 200, ${a})`, lineW);
    };

    const makeColors = (n) => {
      const colors = [];
      for (let i = 0; i < n; i++) {
        const s = 0.5 + (i % 5) * 0.12;
        const l = 0.35 + (i % 4) * 0.12;
        const [r, g, b] = [0.55, l, s].map((v) => Math.round(v * 255));
        colors.push(`rgb(${r},${g},${b})`);
      }
      return colors;
    };

    const scale = (p, imgSize = null) => {
      const w = imgSize?.w > 0 ? imgSize.w : apiW;
      const h = imgSize?.h > 0 ? imgSize.h : apiH;
      const sx = imgW / w;
      const sy = imgH / h;
      return [
        Math.round((p[0] ?? 0) * sx),
        Math.round((p[1] ?? 0) * sy),
        Math.round((p[2] ?? 0) * sx),
        Math.round((p[3] ?? 0) * sy),
      ];
    };

    const drawBoxes = (list, colors, opacity = 1, imageSize = null) => {
      if (!list?.length) return;
      const c = colors.length ? colors : makeColors(list.length);
      for (let i = 0; i < list.length; i++) {
        const [x1, y1, x2, y2] = scale(list[i].position || [], imageSize);
        if (x2 - x1 >= 2 && y2 - y1 >= 2) {
          drawCorner(ctx, x1, y1, x2, y2, c[i % c.length], lineW, opacity);
        }
      }
    };

    const payload = v2PayloadRef.current;
    const trans = v2TransitionRef.current;

    // Уже идёт переход v1→v2: не пересоздаём его из нового payload, очищаем payload и только рисуем
    if (trans) {
      v2PayloadRef.current = null;
      if (trans.rafId === 0 || trans.rafId === undefined) {
        log('draw: transition loop start');
        const runTransitionTick = () => {
          const t = v2TransitionRef.current;
          if (!t) return;
          const n = performance.now();
          const el = n - t.startTime;
          const oldColors = makeColors(t.oldDetections?.length || 0);
          const newColors = makeColors(t.newDetections?.length || 0);

          if (t.phase === 'fadeOut') {
            const prog = Math.min(1, el / V2_FADE_OUT_MS);
            const alpha = 1 - (1 - V2_FADE_OUT_MIN) * prog;
            drawBase();
            drawBoxes(t.oldDetections, oldColors, alpha);
            if (prog >= 1) {
              t.phase = 'fadeIn';
              t.startTime = performance.now();
              log('draw: transition phase fadeIn');
            }
            v2TransitionRef.current.rafId = requestAnimationFrame(runTransitionTick);
            return;
          }
          const prog = Math.min(1, el / V2_FADE_IN_MS);
          const alpha = V2_FADE_IN_START + (1 - V2_FADE_IN_START) * prog;
          drawBase();
          drawBoxes(t.newDetections, newColors, alpha, t.imageSize);
          if (prog >= 1) {
            log('draw: transition done, set v2', { count: t.newDetections?.length });
            setDetections(t.newDetections);
            setResultImageSize(t.imageSize);
            setResultVersion('v2');
            setStatus('done');
            v2TransitionRef.current = null;
            return;
          }
          v2TransitionRef.current.rafId = requestAnimationFrame(runTransitionTick);
        };
        runTransitionTick();
      }
      return;
    }

    // Применяем v2 сразу, если v1 не показывали (опрос принёс сразу done)
    if (payload && resultVersion === null) {
      log('draw: apply v2 immediately (no v1)', { count: payload.detections?.length });
      v2PayloadRef.current = null;
      setDetections(payload.detections);
      setResultImageSize(payload.image);
      setResultVersion('v2');
      setStatus('done');
      drawBase();
      drawBoxes(payload.detections, makeColors(payload.detections?.length || 0), 1, payload.image);
      return;
    }

    // Запуск перехода v1 → v2 один раз (payload уже не трогаем после создания trans)
    if (payload && resultVersion === 'v1') {
      log('draw: start v2 transition', { oldCount: detections?.length, newCount: payload.detections?.length });
      v2PayloadRef.current = null;
      v2TransitionRef.current = {
        phase: 'fadeOut',
        oldDetections: [...(detections || [])],
        newDetections: [...(payload.detections || [])],
        imageSize: payload.image,
        startTime: performance.now(),
        rafId: 0,
      };
      // Запуск цикла в этом же проходе; при следующих проходах trans уже будет, payload очищен выше
      const t = v2TransitionRef.current;
      const runTransitionTick = () => {
        const tt = v2TransitionRef.current;
        if (!tt) return;
        const n = performance.now();
        const el = n - tt.startTime;
        const oldColors = makeColors(tt.oldDetections?.length || 0);
        const newColors = makeColors(tt.newDetections?.length || 0);

        if (tt.phase === 'fadeOut') {
          const prog = Math.min(1, el / V2_FADE_OUT_MS);
          const alpha = 1 - (1 - V2_FADE_OUT_MIN) * prog;
          drawBase();
          drawBoxes(tt.oldDetections, oldColors, alpha);
          if (prog >= 1) {
            tt.phase = 'fadeIn';
            tt.startTime = performance.now();
            log('draw: transition phase fadeIn');
          }
          v2TransitionRef.current.rafId = requestAnimationFrame(runTransitionTick);
          return;
        }
        const prog = Math.min(1, el / V2_FADE_IN_MS);
        const alpha = V2_FADE_IN_START + (1 - V2_FADE_IN_START) * prog;
        drawBase();
        drawBoxes(tt.newDetections, newColors, alpha, tt.imageSize);
        if (prog >= 1) {
          log('draw: transition done, set v2', { count: tt.newDetections?.length });
          setDetections(tt.newDetections);
          setResultImageSize(tt.imageSize);
          setResultVersion('v2');
          setStatus('done');
          v2TransitionRef.current = null;
          return;
        }
        v2TransitionRef.current.rafId = requestAnimationFrame(runTransitionTick);
      };
      t.rafId = requestAnimationFrame(runTransitionTick);
      return;
    }

    const drawDoneAnimated = () => {
      if (doneTimerRef.current) window.clearTimeout(doneTimerRef.current);
      drawBase();
      if (!detections || detections.length === 0) {
        log('draw: v1 fly-in skip (no detections)');
        return;
      }
      log('draw: v1 fly-in start', { count: detections.length });
      const colors = makeColors(detections.length);
      let drawn = 0;
      const drawNext = () => {
        drawBase();
        drawBoxes(detections.slice(0, drawn), colors);
        if (drawn >= detections.length) return;
        const [x1, y1, x2, y2] = scale(detections[drawn].position || []);
        if (x2 - x1 >= 2 && y2 - y1 >= 2) {
          drawCorner(ctx, x1, y1, x2, y2, colors[drawn % colors.length], lineW);
        }
        drawn++;
        if (drawn < detections.length) doneTimerRef.current = window.setTimeout(drawNext, 300);
      };
      doneTimerRef.current = window.setTimeout(drawNext, 300);
    };

    const stopBlink = () => {
      if (blinkRef.current.rafId) {
        cancelAnimationFrame(blinkRef.current.rafId);
        blinkRef.current.rafId = 0;
      }
      blinkRef.current.running = false;
    };

    const startBlink = () => {
      if (blinkRef.current.running) return;
      if (doneTimerRef.current) window.clearTimeout(doneTimerRef.current);
      blinkRef.current.running = true;
      blinkRef.current.startAt = performance.now();
      const tick = (now) => {
        if (!blinkRef.current.running) return;
        const elapsed = Math.max(0, now - blinkRef.current.startAt);
        const t = elapsed % BLINK_CYCLE_MS;
        const k = t < 1000 ? t / 1000 : (t - 1000) / 1000;
        const alpha = t < 1000 ? 1 - (1 - BLINK_MIN_ALPHA) * k : BLINK_MIN_ALPHA + (1 - BLINK_MIN_ALPHA) * k;
        drawBase();
        drawInitialCornerBox(alpha);
        blinkRef.current.rafId = requestAnimationFrame(tick);
      };
      blinkRef.current.rafId = requestAnimationFrame(tick);
    };

    if (status === 'processing' && resultVersion === null) {
      log('draw: branch blink');
      startBlink();
      return;
    }
    if (status === 'processing' && resultVersion === 'v1') {
      stopBlink();
      if (!v1AnimationStartedRef.current) {
        v1AnimationStartedRef.current = true;
        drawDoneAnimated();
      } else {
        drawBase();
        drawBoxes(detections, makeColors(detections?.length || 0));
      }
      return;
    }
    if (status === 'done') {
      log('draw: branch done', { count: detections?.length });
      v1AnimationStartedRef.current = false;
      stopBlink();
      drawBase();
      drawBoxes(detections, makeColors(detections?.length || 0));
      return;
    }

    log('draw: branch idle/other');
    stopBlink();
    drawBase();
  }, [status, detections, resultImageSize, resultVersion, v2Received]);

  return (
    <div className="app">
      <header className="header">
        <h1>HereYouAre</h1>
      </header>
      <main className="main">
        {/* 1) Верхняя область: загрузка */}
        <section
          className="upload-zone"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            const f = e.dataTransfer.files?.[0];
            if (f) onFile({ target: { files: [f] } });
          }}
          onClick={() => document.getElementById('file')?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') document.getElementById('file')?.click();
          }}
        >
          <input type="file" accept="image/*" onChange={onFile} id="file" className="file-input" />
          <div className="upload-label">
            {preview ? 'Сменить изображение' : 'Перетащите файл сюда или нажмите для выбора'}
          </div>
        </section>

        {/* 2) Средняя область: картинка + разметка */}
        {preview && (
          <section className="viewer">
            <div className="media-column">
              <div className="media-stage">
                <div className="media-frame">
                  <canvas ref={canvasRef} className="preview-canvas" />

                  {/* 3) Нижняя область: горизонтальная прокрутка (25% высоты viewer, ширина как у картинки) */}
                  <div className="strip" style={{ height: stripHeightPx ? `${stripHeightPx}px` : undefined }}>
                    <div className="strip-inner">
                      {(detections || []).map((_, i) => (
                        <div key={i} className="strip-item">
                          #{i + 1}
                        </div>
                      ))}
                      {(!detections || detections.length === 0) && (
                        <div className="strip-item muted">Объекты появятся после разметки</div>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              <div className="meta">
                {status === 'uploading' && <div className="status">Загрузка…</div>}
                {status === 'processing' && <div className="status">Обработка…</div>}
                {status === 'done' && <div className="status">Найдено: {detections.length} объектов</div>}
                {error && <div className="error">{error}</div>}
              </div>
            </div>
          </section>
        )}

        {preview && status === 'idle' && (
          <button type="button" className="start-btn" onClick={startJob} disabled={!file}>
            Запустить детекцию
          </button>
        )}
      </main>
    </div>
  );
}
