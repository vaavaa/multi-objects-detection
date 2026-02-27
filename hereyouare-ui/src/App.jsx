import { useState, useCallback, useEffect } from 'react';

const API_BASE = import.meta.env.VITE_HEREYOUARE_URL || 'http://localhost:8082';

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('idle');
  const [detections, setDetections] = useState([]);
  const [error, setError] = useState(null);
  const [imageSize, setImageSize] = useState(null);
  const pollIntervalMs = 250;

  const onFile = useCallback((e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setJobId(null);
    setDetections([]);
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
      setStatus('processing');
    } catch (err) {
      setError(String(err));
      setStatus('idle');
    }
  }, [file]);

  useEffect(() => {
    if (!jobId || status !== 'processing') return;
    const t = setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/v1/job/${jobId}`);
        if (!r.ok) return;
        const data = await r.json();
        if (data.status === 'done') {
          setDetections(data.detections || []);
          setImageSize(data.image);
          setStatus('done');
          return;
        }
        if (data.status === 'failed') {
          setError(data.error || 'Job failed');
          setStatus('idle');
          return;
        }
      } catch (e) {
        setError(String(e));
        setStatus('idle');
      }
    }, pollIntervalMs);
    return () => clearInterval(t);
  }, [jobId, status]);

  const drawCorner = (ctx, x1, y1, x2, y2, color, lineWidth) => {
    const w = x2 - x1;
    const h = y2 - y1;
    const segW = Math.max(1, Math.floor(w * 0.15));
    const segH = Math.max(1, Math.floor(h * 0.15));
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
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
  };

  const [imageDim, setImageDim] = useState({ w: 0, h: 0 });
  const canvasRef = useRef(null);
  const imageRef = useRef(null);

  useEffect(() => {
    if (!preview || !canvasRef.current) return;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = canvasRef.current;
      const imgW = img.naturalWidth;
      const imgH = img.naturalHeight;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = imgW * dpr;
      canvas.height = imgH * dpr;
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      ctx.drawImage(img, 0, 0);
      setImageDim({ w: imgW, h: imgH });

      if (status === 'processing' && animBox.w > 0) {
        const cx = imgW / 2;
        const cy = imgH / 2;
        const side = Math.min(imgW, imgH) * 0.3;
        const x1 = cx - side / 2;
        const y1 = cy - side / 2;
        const x2 = cx + side / 2;
        const y2 = cy + side / 2;
        ctx.strokeStyle = 'rgba(100, 180, 200, 0.9)';
        ctx.lineWidth = Math.max(2, Math.min(imgW, imgH) / 400);
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }

      if (status === 'done' && detections.length > 0) {
        const colors = [];
        for (let i = 0; i < detections.length; i++) {
          const s = 0.5 + (i % 5) * 0.12;
          const l = 0.35 + (i % 4) * 0.12;
          const [r, g, b] = [0.55, l, s].map((v) => Math.round(v * 255));
          colors.push(`rgb(${r},${g},${b})`);
        }
        const lineW = Math.max(2, Math.min(imgW, imgH) / 400);
        let drawn = 0;
        const drawNext = () => {
          ctx.drawImage(img, 0, 0);
          for (let i = 0; i < drawn; i++) {
            const [x1, y1, x2, y2] = (detections[i].position || []).map(Math.round);
            if (x2 - x1 >= 2 && y2 - y1 >= 2) {
              drawCorner(ctx, x1, y1, x2, y2, colors[i % colors.length], lineW);
            }
          }
          if (drawn >= detections.length) return;
          const [x1, y1, x2, y2] = (detections[drawn].position || []).map(Math.round);
          if (x2 - x1 >= 2 && y2 - y1 >= 2) {
            drawCorner(ctx, x1, y1, x2, y2, colors[drawn % colors.length], lineW);
          }
          drawn++;
          if (drawn < detections.length) setTimeout(drawNext, 300);
        };
        setTimeout(drawNext, 300);
      }
    };
    img.src = preview;
  }, [status, preview, detections]);

  return (
    <div className="app">
      <header className="header">
        <h1>HereYouAre</h1>
        <p>Загрузка изображения → асинхронная детекция → опрос 4/с → разметка по углам</p>
      </header>
      <main className="main">
        <div className="upload-zone" onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files?.[0]; if (f) onFile({ target: { files: [f] } }); }}>
          <input type="file" accept="image/*" onChange={onFile} id="file" className="file-input" />
          <label htmlFor="file" className="upload-label">
            {preview ? 'Сменить изображение' : 'Перетащите файл сюда или нажмите для выбора'}
          </label>
        </div>
        {preview && (
          <div className="preview-wrap">
            <img ref={imageRef} src={preview} alt="Preview" className="preview-img" style={{ display: 'none' }} />
            <canvas ref={canvasRef} className="preview-canvas" />
            {status === 'uploading' && <div className="status">Загрузка…</div>}
            {status === 'processing' && <div className="status">Обработка…</div>}
            {status === 'done' && imageDim && (
              <div className="status">Найдено: {detections.length} объектов</div>
            )}
            {error && <div className="error">{error}</div>}
          </div>
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
