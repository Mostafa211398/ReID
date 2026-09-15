import { useEffect, useRef, useState } from "react";
import { Camera, Download, Layers, Plus, RefreshCw, Truck, Trash2, UploadCloud } from "lucide-react";
import { MetricChart } from "../components/MetricChart";
import { reid, type BenchmarkComparison, type BenchmarkComparisonPage, type BenchmarkEncoderResult, type DeletionPreview, type PairComparison, type PairCriteriaAttribute, type PairCriteriaReport, type PairEncoderId, type PairWeights, type ReIDBenchmark, type ReIDComparison, type ReIDComparisonPage, type ReIDScore, type MatchingSettings, type ReIDDataset, type ReIDEncoder, type ReIDEvaluation, type ReIDJob, type ReIDSite, type ReIDTrack } from "../reid";

const pending = (job?: ReIDJob) => job?.state === "queued" || job?.state === "running";
const statusLabel = (status: string) => ({ new: "New vehicle", automatic: "Automatic match", review: "Needs review", reviewed: "Confirmed" }[status] || status);
const percent = (value: number | null) => value === null ? "—" : (value * 100).toFixed(1) + "%";

type BenchmarkUpload = { file: File; relative_path: string; identity: string; source_group: string; valid: boolean; digest: string };

async function inspectFolder(files: File[]): Promise<BenchmarkUpload[]> {
  const rawPaths = files.map(file => file.webkitRelativePath || file.name).map(path => path.replaceAll("\\", "/"));
  const split = rawPaths.map(path => path.split("/").filter(Boolean));
  const stripRoot = split.length > 0 && split.every(parts => parts.length >= 3) && new Set(split.map(parts => parts[0].toLowerCase())).size === 1;
  return Promise.all(files.map(async (file, index) => {
    const parts = stripRoot ? split[index].slice(1) : split[index];
    const valid = parts.length === 2 && /\.(jpe?g|png|webp|bmp)$/i.test(parts[1]);
    const identity = valid ? parts[0] : "";
    const source_group = valid ? parts[1].replace(/\.[^.]+$/, "").toLowerCase() : "";
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", await file.arrayBuffer()))).map(value => value.toString(16).padStart(2, "0")).join("");
    return { file, relative_path: rawPaths[index], identity, source_group, valid, digest };
  }));
}

export function ReIDPage() {
  const [sites, setSites] = useState<ReIDSite[]>([]);
  const [jobs, setJobs] = useState<ReIDJob[]>([]);
  const [encoders, setEncoders] = useState<ReIDEncoder[]>([]);
  const [datasets, setDatasets] = useState<ReIDDataset[]>([]);
  const [evaluations, setEvaluations] = useState<ReIDEvaluation[]>([]);
  const [benchmarks, setBenchmarks] = useState<ReIDBenchmark[]>([]);
  const [pairComparisons, setPairComparisons] = useState<PairComparison[]>([]);
  const [siteId, setSiteId] = useState("");
  const [jobId, setJobId] = useState("");
  const [tracks, setTracks] = useState<ReIDTrack[]>([]);
  const [identities, setIdentities] = useState<string[]>([]);
  const [deletion, setDeletion] = useState<DeletionPreview | null>(null);
  const deletedIds = useRef(new Set<string>());
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [revision, setRevision] = useState(0);
  const [ready, setReady] = useState(false);
  const [section, setSection] = useState<"experiments" | "compare" | "research" | "site">("experiments");
  const [siteName, setSiteName] = useState("");
  const [cameraNames, setCameraNames] = useState("CAM_01\nCAM_02");
  const [files, setFiles] = useState<{ file: File; camera_id: string; start: string; class_name: "" | "car" | "truck" }[]>([]);
  const [experimentName, setExperimentName] = useState("");
  const [selectedEncoders, setSelectedEncoders] = useState(["siglip", "dinov2", "fastreid"]);
  const [refreshEncoders, setRefreshEncoders] = useState<string[]>([]);
  const [encoderId, setEncoderId] = useState("dinov2");
  const [datasetId, setDatasetId] = useState("");
  const [epochs, setEpochs] = useState(20);
  const [focusTrack, setFocusTrack] = useState("");
  const [reviewOnly, setReviewOnly] = useState(false);
  const [selectedExperiments, setSelectedExperiments] = useState<string[]>([]);
  const [benchmarkId, setBenchmarkId] = useState("");
  const [benchmarkFiles, setBenchmarkFiles] = useState<BenchmarkUpload[]>([]);
  const [benchmarkName, setBenchmarkName] = useState("");
  const [benchmarkEncoders, setBenchmarkEncoders] = useState<string[] | null>(null);
  const [benchmarkThreshold, setBenchmarkThreshold] = useState(.85);
  const [pairId, setPairId] = useState("");
  const site = sites.find(s => s.id === siteId);
  const selectedJob = jobs.find(j => j.id === jobId);
  const researchEncoder = encoders.find(e => e.id === encoderId);
  const canTrain = researchEncoder?.supports_training !== false;
  useEffect(() => { setRefreshEncoders(selectedJob?.config.encoders || []); }, [jobId, selectedJob?.config.encoders?.join("|")]);
  const experimentJobs = jobs.filter(j => j.site_id === siteId && j.kind === "experiment");
  const siteDatasets = datasets.filter(d => d.site_id === siteId);
  const comparison = evaluations.filter(e => e.site_id === siteId && (!datasetId || e.dataset_id === datasetId));
  const siteBenchmarks = benchmarks.filter(benchmark => benchmark.site_id === siteId);
  const selectedBenchmark = siteBenchmarks.find(benchmark => benchmark.id === benchmarkId);
  const sitePairComparisons = pairComparisons.filter(comparison => comparison.site_id === siteId);
  const selectedPairComparison = sitePairComparisons.find(comparison => comparison.id === pairId);
  const availableBenchmarkEncoders = encoders.filter(encoder => encoder.origin === "pretrained" && encoder.available);
  const selectedBenchmarkEncoders = benchmarkEncoders ?? availableBenchmarkEncoders.map(encoder => encoder.id);
  const validBenchmarkFiles = benchmarkFiles.filter(item => item.valid);
  const benchmarkIdentities = new Set(validBenchmarkFiles.map(item => item.identity));
  const benchmarkSourceGroups = new Set(validBenchmarkFiles.map(item => item.identity + "\0" + item.source_group));
  const benchmarkInvalid = benchmarkFiles.length - validBenchmarkFiles.length;
  const benchmarkDuplicates = validBenchmarkFiles.length - new Set(validBenchmarkFiles.map(item => item.digest)).size;
  const benchmarkSingletons = [...benchmarkIdentities].filter(identity => validBenchmarkFiles.filter(item => item.identity === identity).length < 2);
  const winner = datasetId ? comparison.filter(e => e.split === "validation" && e.metrics.cross_camera.mAP !== null).sort((a, b) => (b.metrics.cross_camera.mAP || 0) - (a.metrics.cross_camera.mAP || 0))[0] : undefined;
  const history = ((selectedJob?.metrics.train as { history?: { epoch: number; loss: number; mAP: number }[] } | undefined)?.history || []);
  const refresh = () => setRevision(value => value + 1);
  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true); setError("");
    try { await fn(); refresh(); } catch (exc) { setError(String(exc instanceof Error ? exc.message : exc)); }
    finally { setBusy(false); }
  };

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const [s, j, state, d, e, b, p] = await Promise.all([reid.sites(), reid.jobs(), reid.status(), reid.datasets(), reid.evaluations(), reid.benchmarks(), reid.pairComparisons()]);
        if (!alive) return;
        setSites(s); setJobs(j.filter(job => !deletedIds.current.has(job.id))); setEncoders(state.encoders); setReady(state.environment_ready); setDatasets(d); setEvaluations(e); setBenchmarks(b); setPairComparisons(p);
        setSiteId(id => id || s[0]?.id || "");
      } catch (exc) { if (alive) setError(String(exc instanceof Error ? exc.message : exc)); }
    };
    void load();
    const timer = window.setInterval(load, 2000);
    return () => { alive = false; window.clearInterval(timer); };
  }, [revision]);

  useEffect(() => { setBenchmarkThreshold(site?.matching?.threshold ?? .85); }, [siteId]);
  useEffect(() => { if (selectedBenchmark) setBenchmarkThreshold(selectedBenchmark.threshold); }, [benchmarkId, selectedBenchmark?.threshold]);

  useEffect(() => {
    let alive = true;
    setTracks([]); setFocusTrack("");
    const load = async () => {
      try {
        const [t, ids] = await Promise.all([jobId && selectedJob?.kind === "experiment" ? reid.tracks(jobId) : Promise.resolve([]), siteId ? reid.identities(siteId) : Promise.resolve([])]);
        if (alive) { setTracks(t); setIdentities(ids.map(i => i.global_id)); }
      } catch (exc) { if (alive) setError(String(exc instanceof Error ? exc.message : exc)); }
    };
    void load();
    const timer = window.setInterval(load, 3000);
    return () => { alive = false; window.clearInterval(timer); };
  }, [jobId, siteId, selectedJob?.kind, revision]);

  const createSite = () => act(async () => {
    const cameras = cameraNames.split(/\r?\n/).map(s => s.trim()).filter(Boolean).map(id => ({ id, name: id }));
    const value = await reid.createSite({ name: siteName, cameras, transitions: [] });
    setSiteId(value.id); setSiteName(""); setSection("experiments");
  });
  const previewDelete = (id: string) => act(async () => setDeletion(await reid.deletionPreview(id)));
  const selectJob = (job: ReIDJob) => { setJobId(job.id); setSiteId(job.site_id); };
  const visibleTracks = tracks.filter(t => (!reviewOnly || t.assignments[site?.active_encoder || "dinov2"]?.status === "review") && (!focusTrack || t.id === focusTrack));
  return <div className="page-stack reid-workspace">
    <div className="panel reid-intro">
      <div><p className="section-kicker">VEHICLE IDENTITY LAB</p><h2>Recognize vehicles across uploads.</h2><p className="subtle">Upload cropped vehicle images, compare visual encoders, and find saved vehicle identities. Labeling and fine-tuning are optional.</p></div>
      <Truck size={44}/>
    </div>
    {error && <div role="alert" className="notice error">{error}</div>}
    {!ready && <div className="notice warning">ReID setup is required. Run <code>scripts/setup-reid.ps1</code>, then <code>download-reid-models.ps1</code>. You can create sites while setup completes.</div>}
    <div className="reid-toolbar">
      <label><span>Construction site</span><select aria-label="Construction site" value={siteId} onChange={e => { setSiteId(e.target.value); setJobId(""); setBenchmarkId(""); setPairId(""); setFiles([]); setBenchmarkFiles([]); setSelectedExperiments([]); }}><option value="">Choose a site</option>{sites.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}</select></label>
      <div className="button-row">{(["experiments", "compare", "research", "site"] as const).map(id => <button key={id} className={"button " + (section === id ? "primary" : "secondary")} onClick={() => setSection(id)}>{id === "compare" ? "Compare two images" : id === "research" ? "Accuracy & optional training" : id === "site" ? "Site & cameras" : "Experiments"}</button>)}</div>
    </div>

    {(section === "site" || !sites.length) && <div className="panel reid-pad">
      <h3>Create a site</h3><div className="field-row"><label><span>Site name</span><input aria-label="Site name" value={siteName} onChange={e => setSiteName(e.target.value)} placeholder="North construction site"/></label><label><span>Camera IDs · one per line</span><textarea aria-label="Camera IDs" value={cameraNames} onChange={e => setCameraNames(e.target.value)}/></label></div>
      <button className="button primary" disabled={busy || !siteName.trim()} onClick={() => void createSite()}><Plus size={14}/>Create site</button>
      {site && <SiteEditor key={site.id} site={site} encoders={encoders} busy={busy} act={act}/>}
    </div>}

    {section === "experiments" && site && <>
      <div className="reid-encoders">{encoders.filter(e => e.origin === "pretrained").map(e => <div key={e.id} className="panel reid-pad"><Layers size={18}/><strong>{e.name}</strong><span className={e.available ? "reid-ready" : "subtle"}>{e.available ? "Ready · pretrained" : e.availability_reason || "Model download required"}</span><small>{e.runtime} · {e.embedding_dimension} dimensions{e.supports_training === false ? " · Inference & evaluation only" : ""}</small>{e.adaptation && <small>{e.adaptation}</small>}</div>)}</div>
      <div className="panel reid-pad">
        <h3>New crop experiment</h3>
        <p className="subtle">Upload one already cropped vehicle per image. Every selected encoder receives the whole crop. Site IDs use {site.active_encoder}.</p>
        <label><span>Experiment name</span><input aria-label="Experiment name" value={experimentName} onChange={e => setExperimentName(e.target.value)} placeholder="Morning deliveries"/></label>
        <div className="button-row reid-checks">{encoders.map(e => <label key={e.id}><input type="checkbox" checked={selectedEncoders.includes(e.id) || e.id === site.active_encoder} disabled={!e.available || e.id === site.active_encoder} onChange={event => setSelectedEncoders(ids => event.target.checked ? [...ids, e.id] : ids.filter(id => id !== e.id))}/><span>{e.name}</span></label>)}</div>
        <label className="reid-upload"><UploadCloud size={24}/><span>Add vehicle crops</span><input aria-label="Vehicle crops" type="file" multiple accept=".jpg,.jpeg,.png,.webp,.bmp" onChange={e => { setFiles(Array.from(e.target.files || []).map((file, i) => ({ file, camera_id: site.cameras[i % Math.max(1, site.cameras.length)]?.id || "", start: "", class_name: "" }))); e.currentTarget.value = ""; }}/></label>
        {files.map((item, i) => <div className="reid-clip" key={i}><span><UploadThumbnail file={item.file}/>{item.file.name}</span>
          <label><span>Camera</span><select aria-label={"Camera for " + item.file.name} value={item.camera_id} onChange={e => setFiles(items => items.map((v, index) => index === i ? { ...v, camera_id: e.target.value } : v))}>{site.cameras.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}</select></label>
          <label><span>Vehicle class · required</span><select required aria-label={"Vehicle class for " + item.file.name} value={item.class_name} onChange={e => setFiles(items => items.map((v, index) => index === i ? { ...v, class_name: e.target.value as "car" | "truck" } : v))}><option value="">Choose class</option><option value="car">Car</option><option value="truck">Truck</option></select></label>
          <label><span>Capture time · optional</span><input type="datetime-local" value={item.start} onChange={e => setFiles(items => items.map((v, index) => index === i ? { ...v, start: e.target.value } : v))}/></label>
        </div>)}
        <p className="subtle">Leave capture times blank when unknown. Similarity scores are not accuracy percentages.</p>
        <button className="button primary" disabled={busy || !ready || !files.length || !site.cameras.length || !experimentName.trim() || files.some(f => !f.class_name || !f.camera_id)} onClick={() => void act(async () => {
          const job = await reid.upload(files.map(f => f.file), { name: experimentName, site_id: site.id, encoders: Array.from(new Set([site.active_encoder, ...selectedEncoders])), clips: files.map(f => ({ media_type: "image", class_name: f.class_name, camera_id: f.camera_id, start_time: f.start ? new Date(f.start).toISOString() : null })) });
          selectJob(job); setExperimentName(""); setFiles([]);
        })}><UploadCloud size={15}/>{busy ? "Working…" : "Upload & run experiment"}</button>
      </div>
      <div className="panel reid-pad">
        <h3>Experiment history</h3>
        {!experimentJobs.length && <p className="subtle">Your uploaded experiments will appear here.</p>}
        <div className="reid-job-list">{experimentJobs.map(job => <div key={job.id} className="reid-history-row"><button className={"reid-job " + (job.id === jobId ? "selected" : "")} onClick={() => selectJob(job)}><strong>{job.name}</strong><span>{job.deletion_pending ? "Deletion cleanup pending" : job.refresh_required ? "Refresh matching required" : job.state + " · " + job.stage}</span></button><button className="button danger-button" aria-label={"Delete experiment " + job.name} disabled={busy || pending(job)} onClick={() => void previewDelete(job.id)}><Trash2 size={14}/>Delete</button></div>)}</div>
      </div>
    </>}

    {section === "compare" && site && <PairComparisonWorkspace
      site={site}
      pairEncoders={encoders.filter(encoder => ["coca", "coca_visual", "coca_l14", "coca_l14_visual", "siglip2", "siglip2_visual"].includes(encoder.id))}
      comparisons={sitePairComparisons}
      selected={selectedPairComparison}
      busy={busy}
      ready={ready}
      act={act}
      select={comparison => { setPairId(comparison.id); const job = jobs.find(value => value.id === comparison.id); if (job) selectJob(job); }}
      created={job => { setPairId(job.id); selectJob(job); refresh(); }}
      deleted={id => { deletedIds.current.add(id); setPairComparisons(values => values.filter(value => value.id !== id)); setJobs(values => values.filter(value => value.id !== id)); if (pairId === id) setPairId(""); if (jobId === id) setJobId(""); }}
      updated={comparison => setPairComparisons(values => values.map(value => value.id === comparison.id ? comparison : value))}
    />}

    {section === "research" && site && <div className="panel reid-pad">
      <h3>Measure first. Fine-tune only when useful.</h3>
      <p className="subtle">Label observations with the same real vehicle name across cameras. Snapshots preserve labels and split identities into train, validation, and test groups. At least six identities are needed; evaluation also needs matching observations and different vehicles.</p>
      <section className="reid-benchmark">
        <div className="reid-toolbar"><div><h3>Folder benchmark</h3><p className="subtle">Select one parent folder whose immediate child folders are truck identities. Every image is measured with raw embedding cosine.</p></div></div>
        <div className="field-row"><label><span>Benchmark name</span><input aria-label="Benchmark name" value={benchmarkName} onChange={event => setBenchmarkName(event.target.value)} placeholder="Chosen crops benchmark"/></label><label><span>Operational cosine threshold</span><input aria-label="Benchmark threshold" type="number" min={-1} max={1} step={.01} value={benchmarkThreshold} onChange={event => setBenchmarkThreshold(Number(event.target.value))}/></label></div>
        <label className="reid-upload"><UploadCloud size={24}/><span>Select labeled parent folder</span><input aria-label="Labeled truck folder" type="file" multiple accept=".jpg,.jpeg,.png,.webp,.bmp" ref={node => { node?.setAttribute("webkitdirectory", ""); node?.setAttribute("directory", ""); }} onChange={event => { const picked = Array.from(event.currentTarget.files || []); const root = picked[0]?.webkitRelativePath.split(/[\\/]/)[0]; event.currentTarget.value = ""; void inspectFolder(picked).then(items => { setBenchmarkFiles(items); if (!benchmarkName) setBenchmarkName(root || "Folder benchmark"); }); }}/></label>
        {!!benchmarkFiles.length && <div className="benchmark-preflight" aria-label="Benchmark preflight"><span>{validBenchmarkFiles.length} images</span><span>{benchmarkIdentities.size} identities</span><span>{benchmarkSourceGroups.size} viewpoints</span><span>{benchmarkSingletons.length} singleton identities</span><span>{benchmarkInvalid} invalid</span><span>{benchmarkDuplicates} duplicates</span></div>}
        {benchmarkSingletons.length > 0 && <p className="notice warning">{benchmarkSingletons.join(", ")} cannot be identification queries because each has only one image. They remain negative references.</p>}
        {(benchmarkInvalid > 0 || benchmarkDuplicates > 0) && <p role="alert" className="notice error">Remove invalid paths or duplicate image content before running the benchmark.</p>}
        <fieldset className="reid-refresh-encoders"><legend>Embedding models</legend><div className="reid-checks">{encoders.filter(encoder => encoder.origin === "pretrained").map(encoder => <label key={encoder.id}><input type="checkbox" aria-label={"Benchmark with " + encoder.name} checked={selectedBenchmarkEncoders.includes(encoder.id)} disabled={!encoder.available} onChange={event => setBenchmarkEncoders(ids => { const current = ids ?? availableBenchmarkEncoders.map(item => item.id); return event.target.checked ? [...current, encoder.id] : current.filter(id => id !== encoder.id); })}/><span>{encoder.name}</span></label>)}</div></fieldset>
        <button className="button primary" disabled={busy || !ready || !benchmarkName.trim() || validBenchmarkFiles.length < 2 || benchmarkIdentities.size < 2 || ![...benchmarkIdentities].some(identity => validBenchmarkFiles.filter(item => item.identity === identity).length > 1) || benchmarkInvalid > 0 || benchmarkDuplicates > 0 || !selectedBenchmarkEncoders.length || benchmarkThreshold < -1 || benchmarkThreshold > 1} onClick={() => void act(async () => {
          const job = await reid.createBenchmark(validBenchmarkFiles.map(item => item.file), { name: benchmarkName, site_id: site.id, encoders: selectedBenchmarkEncoders, threshold: benchmarkThreshold, items: validBenchmarkFiles.map(item => ({ relative_path: item.relative_path })) });
          setBenchmarkId(job.id); selectJob(job); setBenchmarkFiles([]); setBenchmarkName("");
        })}><Layers size={15}/>{busy ? "Working…" : "Run folder benchmark"}</button>
        <div className="reid-job-list benchmark-history">{siteBenchmarks.map(benchmark => <div className="reid-history-row" key={benchmark.id}><button className={"reid-job " + (benchmark.id === benchmarkId ? "selected" : "")} onClick={() => { setBenchmarkId(benchmark.id); const job = jobs.find(value => value.id === benchmark.id); if (job) selectJob(job); }}><strong>{benchmark.name}</strong><span>{benchmark.state} · {benchmark.counts.images} images · {benchmark.counts.identities} identities</span></button><button aria-label={"Delete benchmark " + benchmark.name} className="button danger-button" disabled={busy || benchmark.state === "queued" || benchmark.state === "running" || ["queued", "running"].includes(benchmark.comparison_export?.state || "")} onClick={() => void act(async () => { await reid.deleteBenchmark(benchmark.id); setBenchmarks(values => values.filter(value => value.id !== benchmark.id)); setJobs(values => values.filter(value => value.id !== benchmark.id && value.config.benchmark_id !== benchmark.id)); if (benchmarkId === benchmark.id) setBenchmarkId(""); if (jobId === benchmark.id) setJobId(""); })}>Delete</button></div>)}</div>
        {selectedBenchmark?.encoders && <BenchmarkResults benchmark={selectedBenchmark} threshold={benchmarkThreshold} setThreshold={setBenchmarkThreshold} busy={busy}
          update={value => void act(async () => { const updated = await reid.updateBenchmarkThreshold(selectedBenchmark.id, value); setBenchmarks(records => records.map(record => record.id === updated.id ? updated : record)); setBenchmarkThreshold(updated.threshold); })}
          prepare={() => void act(() => reid.prepareBenchmarkComparisonImages(selectedBenchmark.id))}
          cancel={id => void act(() => reid.cancel(id))} retry={id => void act(() => reid.retry(id))}/>} 
      </section>
      <div className="reid-checks">{experimentJobs.filter(j => !pending(j)).map(j => <label key={j.id}><input type="checkbox" checked={selectedExperiments.includes(j.id)} onChange={e => setSelectedExperiments(ids => e.target.checked ? [...ids, j.id] : ids.filter(id => id !== j.id))}/><span>{j.name}</span></label>)}</div>
      <button className="button secondary" disabled={busy || !selectedExperiments.length} onClick={() => void act(async () => { const d = await reid.snapshot(site.id, selectedExperiments); setDatasetId(d.id); })}>Create labeled snapshot</button>
      <div className="field-row three reid-spacing"><label><span>Dataset snapshot</span><select aria-label="Dataset snapshot" value={datasetId} onChange={e => setDatasetId(e.target.value)}><option value="">Choose a snapshot</option>{siteDatasets.map(d => <option key={d.id} value={d.id}>{d.id.slice(0, 8)} · {Object.values(d.splits).flat().length} vehicles</option>)}</select></label><label><span>Encoder</span><select aria-label="Research encoder" value={encoderId} onChange={e => setEncoderId(e.target.value)}>{encoders.map(e => <option key={e.id} value={e.id}>{e.name}</option>)}</select></label><label><span>Optional training epochs</span><input type="number" min={1} max={300} value={epochs} onChange={e => setEpochs(Number(e.target.value))}/></label></div>
      <div className="button-row"><button className="button primary" disabled={busy || !datasetId} onClick={() => void act(async () => selectJob(await reid.evaluate(datasetId, encoderId, "validation")))}>Evaluate validation</button><button className="button secondary" disabled={busy || !datasetId} onClick={() => void act(async () => selectJob(await reid.evaluate(datasetId, encoderId, "test")))}>Evaluate held-out test</button><button className="button secondary" disabled={busy || !datasetId || !canTrain || !researchEncoder?.available} onClick={() => void act(async () => selectJob(await reid.train(datasetId, encoderId, epochs)))}>Fine-tune locally · optional</button><button className="button secondary" disabled={busy || !datasetId || !canTrain || !researchEncoder?.available} onClick={() => void act(() => reid.exportTraining(datasetId, encoderId, epochs))}><Download size={14}/>Export training bundle</button></div>
      {!canTrain && <p className="notice">This encoder supports inference and evaluation only. Fine-tuning is not needed to test it.</p>}
      <label className="reid-spacing"><span>Import trained encoder package (.zip)</span><input aria-label="Import encoder" type="file" accept=".zip" disabled={busy} onChange={e => { const file = e.target.files?.[0]; if (file) void act(() => reid.importEncoder(file)); }}/></label>
      <h3>Accuracy comparison</h3><p className="subtle">Compare the same snapshot, split, and color weight. Original Rank-1/mAP columns measure encoder appearance only; combined metrics and false merges use appearance plus color. Validation selects models and thresholds; the test split measures the final result.</p>
      {winner && <p className="notice">Best validation mAP in this snapshot: {encoders.find(e => e.id === winner.encoder_id)?.name || winner.encoder_id} · {percent(winner.metrics.cross_camera.mAP)}. Evaluate the held-out test split before selecting your site encoder.</p>}
      <div className="reid-table"><table><thead><tr><th>Encoder / split / snapshot</th><th>Cross-camera Rank-1</th><th>Rank-5</th><th>mAP</th><th>Same-camera Rank-1</th><th>Combined cross-camera Rank-1 / mAP</th><th>Queries</th><th>False merges / missed</th></tr></thead><tbody>{comparison.map(e => <tr key={e.id}><td>{encoders.find(c => c.id === e.encoder_id)?.name || e.encoder_id}<small>{e.split} · {e.dataset_id.slice(0, 8)}</small></td><td>{percent(e.metrics.cross_camera.rank1)}</td><td>{percent(e.metrics.cross_camera.rank5)}</td><td>{percent(e.metrics.cross_camera.mAP)}</td><td>{percent(e.metrics.same_camera.rank1)}</td><td>{e.metrics.combined_cross_camera ? percent(e.metrics.combined_cross_camera.rank1) + " / " + percent(e.metrics.combined_cross_camera.mAP) : "—"}<small>{e.metrics.scoring ? "Color weight " + e.metrics.scoring.color_weight + " · " + e.metrics.scoring.fingerprint.slice(0, 8) : "Legacy appearance scoring"}</small></td><td>{e.metrics.cross_camera.eligible_queries} eligible / {e.metrics.cross_camera.excluded_queries} excluded</td><td>{e.metrics.false_merges} / {e.metrics.missed_matches}</td></tr>)}</tbody></table></div>
      <div className="reid-job-list">{jobs.filter(j => j.site_id === site.id && j.kind !== "experiment" && j.kind !== "benchmark" && j.kind !== "benchmark_export").map(j => <button className="reid-job" key={j.id} onClick={() => selectJob(j)}><strong>{j.name}</strong><span>{j.state}</span></button>)}</div>
    </div>}

    {selectedJob && selectedJob.kind !== "comparison" && selectedJob.site_id === siteId && <div className="panel reid-pad">
      <div className="reid-toolbar"><div><h3>{selectedJob.name}</h3><p className="subtle">{selectedJob.state} · {selectedJob.stage}</p></div><div className="button-row">{pending(selectedJob) ? <button className="button secondary" disabled={busy} onClick={() => void act(() => reid.cancel(selectedJob.id))}>Cancel</button> : <button className="button secondary" disabled={busy || selectedJob.deletion_pending} onClick={() => void act(() => reid.retry(selectedJob.id, selectedJob.kind === "experiment" ? refreshEncoders : undefined))}><RefreshCw size={14}/>{selectedJob.state === "completed" && selectedJob.kind === "experiment" ? "Refresh matching & results" : "Retry cached job"}</button>}</div></div>
      {selectedJob.kind === "experiment" && <button className="button danger-button" disabled={busy || pending(selectedJob)} onClick={() => void previewDelete(selectedJob.id)}><Trash2 size={14}/>Delete experiment</button>}
      {selectedJob.kind === "experiment" && !pending(selectedJob) && <fieldset className="reid-refresh-encoders" disabled={busy || selectedJob.deletion_pending}><legend>Encoders for refresh</legend><div className="reid-checks">{encoders.map(e => <label key={e.id}><input type="checkbox" aria-label={"Refresh with " + e.name} checked={refreshEncoders.includes(e.id) || e.id === site?.active_encoder} disabled={!e.available || e.id === site?.active_encoder} onChange={event => setRefreshEncoders(ids => event.target.checked ? [...ids, e.id] : ids.filter(id => id !== e.id))}/><span>{e.name}</span></label>)}</div><small>Refresh reuses saved crops and embeddings. Include the new encoder in earlier reference experiments too, or rebuild the site gallery with it.</small></fieldset>}
      {selectedJob.refresh_required && <p className="notice warning">Refresh matching required. A previous reference was deleted; source media and embeddings are preserved.</p>}
      {selectedJob.deletion_error && <p role="alert" className="notice error">{selectedJob.deletion_error}</p>}
      {Object.entries(selectedJob.metrics).map(([stage, value]) => { const fallback = (value as { fallback?: string } | null)?.fallback; return fallback ? <p className="notice" key={stage}>{fallback}</p> : null; })}
      {pending(selectedJob) && <progress max={1} value={selectedJob.progress}/>}
      {selectedJob.error && <div className="notice error">{selectedJob.error}<br/><a href={"/api/reid/jobs/" + selectedJob.id + "/artifacts/" + selectedJob.stage + ".log"}>Download stage log</a></div>}
      {selectedJob.kind === "experiment" && selectedJob.state === "completed" && !selectedJob.refresh_required && !selectedJob.deletion_pending && <a className="button primary" href={"/api/reid/experiments/" + selectedJob.id + "/results.zip"}><Download size={14}/>Download all results (.zip)</a>}
      <div className="button-row">{Object.entries(selectedJob.artifacts).filter(([name]) => name !== "Complete results (.zip)").map(([name, url]) => <a key={name} className="button secondary" href={url}><Download size={14}/>{name}</a>)}</div>
      {history.length > 0 && <div className="field-row reid-spacing"><MetricChart values={history.map(h => h.loss)} label="Training loss"/><MetricChart values={history.map(h => h.mAP)} label="Validation mAP" color="#69a7f6"/></div>}
      <div className="button-row reid-spacing">{Object.entries(selectedJob.metrics).filter(([, value]) => typeof value === "object" && value !== null && "seconds" in value).map(([stage, value]) => { const metric = value as { seconds: number; peak_vram_mb?: number }; return <span className="chip" key={stage}>{stage.split("-").slice(0, 2).join(" ")} · {metric.seconds.toFixed(1)}s{metric.peak_vram_mb !== undefined ? " · " + metric.peak_vram_mb.toFixed(0) + " MB peak VRAM" : ""}</span>; })}</div>
      <ImageResults job={selectedJob}/>
      {selectedJob.kind === "experiment" && selectedJob.state === "completed" && !tracks.length && <p className="notice">No observations available. Check the experiment error and retry.</p>}
      {selectedJob.kind === "experiment" && <><div className="reid-toolbar reid-spacing"><h3>Vehicle observations · {tracks.length}</h3><div className="button-row"><button className="button secondary" onClick={() => setReviewOnly(v => !v)}>{reviewOnly ? "Show all observations" : "Show review queue"}</button>{focusTrack && <button className="button secondary" onClick={() => setFocusTrack("")}>Clear observation filter</button>}</div></div><p className="subtle">Assign site IDs to confirm matches. Optional true-identity labels are used for evaluation only. Refresh matching and results after review to update exports.</p><div className="reid-track-grid">{visibleTracks.map(track => <TrackCard key={track.id} track={track} identities={identities} encoders={encoders} activeEncoder={site?.active_encoder || "dinov2"} comparisonReady={selectedJob.state === "completed" && !selectedJob.refresh_required && !selectedJob.deletion_pending} disabled={busy || pending(selectedJob) || !!selectedJob.deletion_pending} act={act} focus={setFocusTrack}/>)}</div>
        {selectedJob.state === "completed" && !selectedJob.refresh_required && !selectedJob.deletion_pending && <ComparisonBrowser experimentId={selectedJob.id} defaultEncoder={site?.active_encoder || "dinov2"} title="Experiment image comparison table"/>}</>}
    </div>}
    {deletion && <DeleteExperimentDialog preview={deletion} busy={busy} error={error} close={() => setDeletion(null)} confirm={() => void act(async () => {
      await reid.deleteExperiment(deletion.experiment_id);
      deletedIds.current.add(deletion.experiment_id);
      setJobs(values => values.filter(j => j.id !== deletion.experiment_id));
      if (jobId === deletion.experiment_id) { setJobId(""); setTracks([]); setFocusTrack(""); }
      setSelectedExperiments(values => values.filter(id => id !== deletion.experiment_id));
      setIdentities([]); setDeletion(null);
    })}/>}
  </div>;
}

const pairWeightKeys: (keyof PairWeights)[] = ["appearance", "color", "shape", "semantic"];

function PairComparisonWorkspace({ site, pairEncoders, comparisons, selected, busy, ready, act, select, created, deleted, updated }: {
  site: ReIDSite; pairEncoders: ReIDEncoder[]; comparisons: PairComparison[]; selected?: PairComparison; busy: boolean; ready: boolean;
  act: (fn: () => Promise<unknown>) => Promise<void>; select: (comparison: PairComparison) => void;
  created: (job: ReIDJob) => void; deleted: (id: string) => void; updated: (comparison: PairComparison) => void;
}) {
  const [name, setName] = useState("VLM vehicle comparison");
  const [encoderId, setEncoderId] = useState<PairEncoderId>("coca");
  const [files, setFiles] = useState<(File | null)[]>([null, null]);
  const [threshold, setThreshold] = useState(.75);
  const [weights, setWeights] = useState<PairWeights>({ appearance: .60, color: .20, shape: .15, semantic: .05 });
  useEffect(() => {
    if (!selected) return;
    setThreshold(selected.threshold); setWeights(selected.weights); setEncoderId(selected.encoder_id);
  }, [selected?.id, selected?.updated_at]);
  const weightTotal = pairWeightKeys.reduce((total, key) => total + weights[key], 0);
  const validWeights = Math.abs(weightTotal - 1) <= 1e-6;
  const selectedEncoder = pairEncoders.find(encoder => encoder.id === encoderId);
  const encoderOnly = selectedEncoder?.pair_mode === "visual_cosine" || encoderId.endsWith("_visual");
  const result = selected?.result;
  const resultModelLabel = result?.encoder.family.startsWith("coca_l14") ? "CoCa L/14" : result?.encoder.family.startsWith("coca") ? "CoCa" : "SigLIP2";
  const setFile = (index: number, file: File | null) => setFiles(values => values.map((value, position) => position === index ? file : value));
  const score = (value: number | null | undefined) => value == null ? "Unavailable" : value.toFixed(4);
  return <div className="panel reid-pad pair-workspace">
    <div className="reid-toolbar"><div><p className="section-kicker">VLM VECTOR LAB</p><h2>Compare two vehicle images.</h2><p className="subtle">Choose CoCa or SigLIP2 for appearance, color, edge shape, and prompt-based semantic fusion, or use either standalone visual tower for raw cosine.</p></div><Layers size={36}/></div>
    {!selectedEncoder?.available && <p className="notice warning">The selected encoder is unavailable. Run <code>scripts/download-reid-models.ps1 -Encoders {encoderId}</code>{selectedEncoder?.availability_reason ? " · " + selectedEncoder.availability_reason : ""}</p>}
    <label><span>Comparison name</span><input aria-label="Pair comparison name" value={name} maxLength={100} onChange={event => setName(event.target.value)}/></label>
    <label><span>Encoder mode</span><select aria-label="Pair encoder mode" value={encoderId} onChange={event => setEncoderId(event.target.value as PairEncoderId)}><option value="coca">Full CoCa B/32 criteria and fusion</option><option value="coca_visual">CoCa B/32 visual encoder only · raw cosine</option><option value="coca_l14">Full CoCa L/14 criteria and fusion</option><option value="coca_l14_visual">CoCa L/14 visual encoder only · raw cosine</option><option value="siglip2">Full SigLIP2 criteria and fusion</option><option value="siglip2_visual">SigLIP2 visual encoder only · raw cosine</option></select></label>
    <div className="pair-upload-grid">{[0, 1].map(index => <label className="reid-upload pair-upload" key={index}><UploadCloud size={24}/><span>{index === 0 ? "First vehicle image" : "Second vehicle image"}</span><input aria-label={index === 0 ? "First comparison image" : "Second comparison image"} type="file" accept=".jpg,.jpeg,.png,.webp,.bmp" onChange={event => { setFile(index, event.currentTarget.files?.[0] || null); event.currentTarget.value = ""; }}/>{files[index] && <><UploadThumbnail file={files[index]!}/><strong>{files[index]!.name}</strong></>}</label>)}</div>
    {!encoderOnly && <div className="pair-scoring-panel">
      <div className="reid-toolbar"><h3>Combined-score controls</h3><span className={validWeights ? "chip" : "chip pair-invalid"}>{(weightTotal * 100).toFixed(0)}% total</span></div>
      <div className="pair-weight-grid">{pairWeightKeys.map(key => <label key={key}><span>{key[0].toUpperCase() + key.slice(1)} weight</span><input aria-label={key + " comparison weight"} type="number" min={0} max={100} step={1} value={Math.round(weights[key] * 100)} onChange={event => setWeights(value => ({ ...value, [key]: Number(event.target.value) / 100 }))}/><small>{Math.round(weights[key] * 100)}%</small></label>)}<label><span>Same-vehicle threshold</span><input aria-label="Pair comparison threshold" type="number" min={0} max={1} step={.01} value={threshold} onChange={event => setThreshold(Number(event.target.value))}/><small>Experimental and uncalibrated</small></label></div>
      {!validWeights && <p role="alert" className="notice error">Appearance, color, shape, and semantic weights must add up to 100%.</p>}
    </div>}
    {encoderOnly && <div className="pair-scoring-panel"><label><span>Raw cosine threshold</span><input aria-label="Pair comparison threshold" type="number" min={-1} max={1} step={.01} value={threshold} onChange={event => setThreshold(Number(event.target.value))}/><small>Only the two normalized {selectedEncoder?.embedding_dimension ?? "model"}D visual embeddings are used.</small></label></div>}
    <div className="button-row"><button className="button primary" disabled={busy || !ready || !selectedEncoder?.available || files.some(file => !file) || !name.trim() || (!encoderOnly && !validWeights) || threshold < (encoderOnly ? -1 : 0) || threshold > 1} onClick={() => void act(async () => {
      const job = await reid.createPairComparison(files as File[], { name, site_id: site.id, encoder_id: encoderId, threshold, weights: encoderOnly ? { appearance: 1, color: 0, shape: 0, semantic: 0 } : weights });
      created(job); setFiles([null, null]); setName("VLM vehicle comparison");
    })}><Layers size={15}/>{busy ? "Working…" : "Embed and compare"}</button>{selected?.state === "completed" && <button className="button secondary" disabled={busy || (!encoderOnly && !validWeights)} onClick={() => void act(async () => updated(await reid.updatePairScoring(selected.id, threshold, encoderOnly ? { appearance: 1, color: 0, shape: 0, semantic: 0 } : weights)))}>Apply scoring without re-embedding</button>}</div>
    <div className="pair-history reid-spacing"><h3>Saved comparisons</h3>{!comparisons.length && <p className="subtle">Completed and in-progress comparisons for this site appear here.</p>}<div className="reid-job-list">{comparisons.map(comparison => <div className="reid-history-row" key={comparison.id}><button className={"reid-job " + (comparison.id === selected?.id ? "selected" : "")} onClick={() => select(comparison)}><strong>{comparison.name}</strong><span>{comparison.state} · {comparison.stage}</span></button><button className="button danger-button" aria-label={"Delete comparison " + comparison.name} disabled={busy || comparison.state === "queued" || comparison.state === "running"} onClick={() => void act(async () => { await reid.deletePairComparison(comparison.id); deleted(comparison.id); })}><Trash2 size={14}/>Delete</button></div>)}</div></div>
    {selected && <section className="pair-result reid-spacing">
      <div className="reid-toolbar"><div><h3>{selected.name}</h3><p className="subtle">{selected.state} · {selected.stage}</p></div><div className="button-row">{(selected.state === "queued" || selected.state === "running") && <button className="button secondary" disabled={busy} onClick={() => void act(() => reid.cancel(selected.id))}>Cancel</button>}{(selected.state === "failed" || selected.state === "cancelled") && <button className="button secondary" disabled={busy} onClick={() => void act(() => reid.retry(selected.id))}><RefreshCw size={14}/>Retry cached job</button>}{selected.state === "completed" && <a className="button primary" href={"/api/reid/pair-comparisons/" + selected.id + "/results.json"}><Download size={14}/>Download vectors JSON</a>}</div></div>
      {(selected.state === "queued" || selected.state === "running") && <progress max={1} value={selected.progress}/>} {selected.error && <p role="alert" className="notice error">{selected.error}</p>}
      {result && <>
        <div className="pair-decision"><span>{result.encoder_only ? "Raw visual cosine" : "Combined similarity"}</span><strong>{score(result.similarities.combined)}</strong><span className={result.decision.same_vehicle ? "chip pair-same" : "chip pair-different"}>{result.decision.same_vehicle ? "Same vehicle" : "Different vehicles"} · experimental</span></div>
        {result.encoder_only ? <div className="pair-score-grid"><div><span>Raw {resultModelLabel} visual cosine</span><strong>{score(result.similarities.appearance_cosine_raw)}</strong><small>Standalone {result.encoder.embedding_dimension}D visual tower only</small></div></div> : <div className="pair-score-grid">{pairWeightKeys.map(key => <div key={key}><span>{key === "appearance" ? "Appearance similarity" : key + " similarity"}</span><strong>{score(result.similarities[key])}</strong><small>Effective weight {(result.scoring.effective_weights[key] * 100).toFixed(0)}%{!result.availability[key] ? " · unavailable" : ""}</small></div>)}<div><span>Raw {resultModelLabel} cosine</span><strong>{score(result.similarities.appearance_cosine_raw)}</strong><small>{result.encoder.embedding_dimension}D normalized vectors</small></div></div>}
        <div className="pair-result-images">{selected.images.map(image => { const details = result.images.find(value => value.side === image.side); return <article key={image.side}><a href={image.image_url} target="_blank" rel="noreferrer"><img src={image.image_url} alt={(image.side === "first" ? "First" : "Second") + " pair comparison image"}/></a><h4>{image.filename}</h4><p className="subtle">Appearance norm {score(details?.norms.appearance)}{!result.encoder_only && <> · color {details?.color_method || "unavailable"}</>}</p>{!result.encoder_only && <details><summary>{resultModelLabel} semantic attributes</summary>{details && Object.entries(details.semantic_attributes).map(([group, attribute]) => <div className="pair-attribute" key={group}><span>{group.replaceAll("_", " ")}</span><strong>{attribute.label}</strong><small>{(attribute.confidence * 100).toFixed(1)}%</small></div>)}</details>}</article>; })}</div>
        {result.criteria_report && <CriteriaReportPanel report={result.criteria_report}/>} 
        <p className="subtle">Encoder {result.encoder.name} · fingerprint <code>{result.encoder.fingerprint.slice(0, 16)}</code> · decision threshold {result.decision.threshold.toFixed(2)}. Brand labels are constrained diagnostic estimates; exact model lettering and plate characters require OCR and remain unknown.</p>
      </>}
    </section>}
  </div>;
}

function CriteriaAttributeRow({ label, value }: { label: string; value: PairCriteriaAttribute }) {
  return <div className="criteria-attribute"><strong>{label.replaceAll("_", " ")}</strong><span>Image 1: {value.image_1.label} · {value.image_1.confidence == null ? "unavailable" : (value.image_1.confidence * 100).toFixed(1) + "%"}</span><span>Image 2: {value.image_2.label} · {value.image_2.confidence == null ? "unavailable" : (value.image_2.confidence * 100).toFixed(1) + "%"}</span><small>Consistency {value.consistency_score.toFixed(3)}</small></div>;
}

function CriteriaReportPanel({ report }: { report: PairCriteriaReport }) {
  const attributes = [
    ["cargo structure", report.appearance_embedding.cargo_structure], ["cage condition", report.appearance_embedding.cage_condition],
    ["barrel wrapping", report.appearance_embedding.barrel_wrapping], ["human presence", report.appearance_embedding.human_presence],
    ["truck outline", report.shape_extractor.truck_outline], ["cage grid pattern", report.shape_extractor.cage_grid_pattern],
    ["barrel shape", report.shape_extractor.barrel_shape], ["spatial layout", report.shape_extractor.spatial_layout],
    ["brand / logo", report.plate_logo_type_viewpoint.brand_logo], ["logo visibility", report.plate_logo_type_viewpoint.logo_visibility], ["vehicle type", report.plate_logo_type_viewpoint.vehicle_type],
    ["viewpoint", report.plate_logo_type_viewpoint.viewpoint], ["human activity", report.plate_logo_type_viewpoint.human_activity],
  ].filter((value): value is [string, PairCriteriaAttribute] => Boolean(value[1] && typeof value[1] === "object" && "consistency_score" in (value[1] as object)));
  const metric = (value: unknown) => typeof value === "number" ? value.toFixed(3) : "Unavailable";
  return <section className="criteria-report"><div className="reid-toolbar"><div><p className="section-kicker">STRUCTURED CRITERIA · {report.version}</p><h3>Criteria similarity {report.criteria_similarity_score.toFixed(4)}</h3></div><span className={"chip criteria-" + report.qualitative_band}>{report.qualitative_band} · experimental</span></div>
    <div className="pair-score-grid"><div><span>Shape</span><strong>{metric(report.appearance_embedding.shape_similarity)}</strong></div><div><span>Texture</span><strong>{metric(report.appearance_embedding.texture_consistency)}</strong></div><div><span>Detail</span><strong>{metric(report.appearance_embedding.detail_preservation)}</strong></div><div><span>Color variance</span><strong>{metric(report.color_extractor.color_variance)}</strong></div><div><span>Hue shift</span><strong>{metric(report.color_extractor.hue_shift)}</strong></div><div><span>View angle difference</span><strong>{metric(report.shape_extractor.viewpoint_angle_diff)}°</strong></div><div><span>Scale ratio</span><strong>{metric(report.shape_extractor.scale_ratio)}</strong></div></div>
    <div className="criteria-colors"><strong>Dominant colors</strong><span>Image 1: {report.color_extractor.dominant_colors.image_1.join(", ") || "unavailable"}</span><span>Image 2: {report.color_extractor.dominant_colors.image_2.join(", ") || "unavailable"}</span><span>Shared: {report.color_extractor.dominant_colors.shared.join(", ") || "none"}</span></div>
    <div className="criteria-grid">{attributes.map(([label, value]) => <CriteriaAttributeRow key={label} label={label} value={value}/>)}</div>
    <p className="subtle">Exact model text and license-plate characters are reported as unknown because OCR is unavailable. People, background, and viewpoint are diagnostic and do not lower the criteria score.</p>
  </section>;
}

function BenchmarkResults({ benchmark, threshold, setThreshold, busy, update, prepare, cancel, retry }: { benchmark: ReIDBenchmark; threshold: number; setThreshold: (value: number) => void; busy: boolean; update: (value: number) => void; prepare: () => void; cancel: (id: string) => void; retry: (id: string) => void }) {
  const exportState = benchmark.comparison_export;
  const exportPending = exportState?.state === "queued" || exportState?.state === "running";
  const overLimit = Boolean(exportState && exportState.comparison_count > exportState.limit);
  return <div className="benchmark-results">
    <div className="reid-toolbar"><div><h3>{benchmark.name} results</h3><p className="subtle">Raw cosine only · {benchmark.counts.images} images · {benchmark.counts.identities} identities · {benchmark.counts.all_pairs_per_encoder ?? 0} pairs per model</p></div><a className="button primary" href={"/api/reid/benchmarks/" + benchmark.id + "/results.zip"}><Download size={14}/>Download benchmark (.zip)</a></div>
    <section className="benchmark-comparison-export">
      <div className="reid-toolbar"><div><h4>Model comparison images</h4><p className="subtle">{exportState?.comparison_count ?? 0} JPEG cards · every scored pair grouped into a folder named for its model · no re-embedding</p></div><div className="button-row">
        {exportState?.ready ? <a className="button primary" href={"/api/reid/benchmarks/" + benchmark.id + "/comparison-images.zip"}><Download size={14}/>Download model comparison images (.zip)</a>
          : exportPending && exportState.job_id ? <button className="button secondary" disabled={busy} onClick={() => cancel(exportState.job_id!)}>Cancel comparison export</button>
          : exportState?.state === "failed" || exportState?.state === "cancelled" ? <button className="button secondary" disabled={busy || !exportState.job_id} onClick={() => exportState.job_id && retry(exportState.job_id)}><RefreshCw size={14}/>Retry comparison export</button>
          : <button className="button primary" disabled={busy || overLimit} onClick={prepare}><Layers size={14}/>Prepare comparison images</button>}
      </div></div>
      {exportPending && <><progress aria-label="Comparison image export progress" max={1} value={exportState?.progress || 0}/><p className="subtle">Rendering comparison cards in the background… {Math.round((exportState?.progress || 0) * 100)}%</p></>}
      {overLimit && <p role="alert" className="notice warning">This export exceeds the {exportState?.limit.toLocaleString()}-image limit. Use a smaller benchmark folder.</p>}
      {exportState?.error && <p role="alert" className="notice error">{exportState.error}</p>}
    </section>
    <div className="field-row benchmark-threshold"><label><span>Recalculate operational threshold</span><input aria-label="Results threshold" type="number" min={-1} max={1} step={.01} value={threshold} onChange={event => setThreshold(Number(event.target.value))}/></label><button className="button secondary" disabled={busy || exportPending || threshold < -1 || threshold > 1 || threshold === benchmark.threshold} onClick={() => update(threshold)}>Apply without re-embedding</button></div>
    <p className="subtle">Precision, recall, F1 and pair accuracy use the configured threshold. Identification uses the nearest eligible viewpoint. Best F1 is exploratory because its threshold was selected on these same images.</p>
    <div className="reid-table benchmark-summary"><table><thead><tr><th>Embedding model</th><th>Threshold</th><th>Pair precision</th><th>Pair recall</th><th>Pair F1</th><th>Pair accuracy</th><th>ID macro precision / recall / F1</th><th>ID accuracy</th><th>Rank-1 / mAP</th><th>Best F1 threshold</th></tr></thead><tbody>{benchmark.encoders?.map(result => <tr key={result.encoder_id}><td><strong>{result.encoder_name}</strong><small>{result.embedding_dimension} dimensions · {result.encoder_fingerprint.slice(0, 10)}</small></td><td>{result.configured_threshold.toFixed(3)}</td><td>{percent(result.pairwise_strict.precision)}</td><td>{percent(result.pairwise_strict.recall)}</td><td>{percent(result.pairwise_strict.f1)}</td><td>{percent(result.pairwise_strict.accuracy)}</td><td>{percent(result.identification.macro_precision)} / {percent(result.identification.macro_recall)} / {percent(result.identification.macro_f1)}</td><td>{percent(result.identification.accuracy)}</td><td>{percent(result.identification.rank1)} / {percent(result.identification.mAP)}</td><td>{result.best_f1_exploratory.threshold.toFixed(3)}<small>{percent(result.best_f1_exploratory.f1)} F1</small></td></tr>)}</tbody></table></div>
    <BenchmarkComparisonBrowser benchmark={benchmark}/>
    {benchmark.encoders?.map(result => <BenchmarkDetails key={result.encoder_id} result={result}/>)}
  </div>;
}

function BenchmarkDetails({ result }: { result: BenchmarkEncoderResult }) {
  const identities = (result.per_identity || []).map(row => row.identity);
  const confusion = new Map((result.confusion || []).map(row => [row.actual_identity + "\0" + row.predicted_identity, row.count]));
  return <details className="benchmark-details"><summary>{result.encoder_name} details · TP {result.pairwise_strict.tp}, FP {result.pairwise_strict.fp}, TN {result.pairwise_strict.tn}, FN {result.pairwise_strict.fn}</summary>
    <h4>Per-identity identification</h4><div className="reid-table"><table><thead><tr><th>Identity</th><th>Queries</th><th>TP / FP / TN / FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{(result.per_identity || []).map(row => <tr key={row.identity}><td>{row.identity}{!row.included_in_macro && <small>Excluded from macro · no second viewpoint</small>}</td><td>{row.support}</td><td>{row.tp} / {row.fp} / {row.tn} / {row.fn}</td><td>{percent(row.precision)}</td><td>{percent(row.recall)}</td><td>{percent(row.f1)}</td></tr>)}</tbody></table></div>
    <h4>Identification confusion matrix</h4><div className="reid-table confusion-table"><table><thead><tr><th>Actual ↓ / predicted →</th>{identities.map(identity => <th key={identity}>{identity}</th>)}</tr></thead><tbody>{identities.map(actual => <tr key={actual}><th>{actual}</th>{identities.map(predicted => <td key={predicted}>{confusion.get(actual + "\0" + predicted) || 0}</td>)}</tr>)}</tbody></table></div>
    <p className="subtle">All-image diagnostic: precision {percent(result.pairwise_all_images.precision)}, recall {percent(result.pairwise_all_images.recall)}, F1 {percent(result.pairwise_all_images.f1)}, accuracy {percent(result.pairwise_all_images.accuracy)}.</p>
  </details>;
}

function BenchmarkComparisonBrowser({ benchmark }: { benchmark: ReIDBenchmark }) {
  const [open, setOpen] = useState(false);
  const [page, setPage] = useState<BenchmarkComparisonPage | null>(null);
  const [encoderId, setEncoderId] = useState(benchmark.encoders?.[0]?.encoder_id || "");
  const [identity, setIdentity] = useState("");
  const [pairType, setPairType] = useState<"all" | "same" | "different">("all");
  const [eligibility, setEligibility] = useState<"all" | "strict" | "excluded">("strict");
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");
  const limit = 24;
  useEffect(() => { setEncoderId(benchmark.encoders?.[0]?.encoder_id || ""); setOffset(0); setPage(null); }, [benchmark.id, benchmark.updated_at]);
  useEffect(() => {
    if (!open) return;
    let alive = true;
    setLoading(true); setLoadError("");
    void reid.benchmarkComparisons(benchmark.id, { encoder_id: encoderId || undefined, identity: identity || undefined, pair_type: pairType, eligibility, offset, limit })
      .then(value => { if (alive) setPage(value); })
      .catch(exc => { if (alive) setLoadError(String(exc instanceof Error ? exc.message : exc)); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [open, benchmark.id, benchmark.updated_at, encoderId, identity, pairType, eligibility, offset]);
  const reset = (change: () => void) => { setOffset(0); change(); };
  return <section className="comparison-browser benchmark-comparison-browser">
    <button className="button primary comparison-toggle" aria-expanded={open} onClick={() => setOpen(value => !value)}>{open ? "Hide compared images" : "View compared images"}</button>
    {open && <div className="comparison-body">
      <div className="reid-toolbar"><div><h3>Benchmark image comparisons</h3><p className="subtle">Every unordered image pair scored by the selected embedding model, sorted by raw cosine similarity.</p></div>{page && <span className="chip">{page.total} pairs</span>}</div>
      <div className="comparison-filters">
        <label><span>Encoder</span><select aria-label="Benchmark comparison encoder" value={encoderId} onChange={event => reset(() => setEncoderId(event.target.value))}><option value="">All encoders</option>{page?.available_encoders.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
        <label><span>Vehicle folder</span><select aria-label="Benchmark comparison identity" value={identity} onChange={event => reset(() => setIdentity(event.target.value))}><option value="">All identities</option>{page?.available_identities.map(value => <option key={value}>{value}</option>)}</select></label>
        <label><span>Ground truth</span><select aria-label="Benchmark pair type" value={pairType} onChange={event => reset(() => setPairType(event.target.value as "all" | "same" | "different"))}><option value="all">Same and different trucks</option><option value="same">Same truck only</option><option value="different">Different trucks only</option></select></label>
        <label><span>Metric eligibility</span><select aria-label="Benchmark pair eligibility" value={eligibility} onChange={event => reset(() => setEligibility(event.target.value as "all" | "strict" | "excluded"))}><option value="strict">Strict metric pairs</option><option value="excluded">Excluded same-source pairs</option><option value="all">All pairs</option></select></label>
      </div>
      {loadError && <p role="alert" className="notice error">{loadError}</p>}
      {loading && <p className="subtle comparison-loading">Loading benchmark image pairs…</p>}
      {!loading && page?.total === 0 && <p className="notice">No benchmark image pairs match these filters.</p>}
      {!loading && page && page.items.length > 0 && <div className="benchmark-pair-grid">{page.items.map((row, index) => <BenchmarkPair key={row.encoder_id + row.left_id + row.right_id + index} row={row} configuredThreshold={benchmark.threshold}/>)}</div>}
      {page && page.total > 0 && <div className="comparison-pagination"><span>{offset + 1}–{Math.min(offset + limit, page.total)} of {page.total}</span><div className="button-row"><button className="button secondary" disabled={loading || offset === 0} onClick={() => setOffset(value => Math.max(0, value - limit))}>Previous</button><button className="button secondary" disabled={loading || offset + limit >= page.total} onClick={() => setOffset(value => value + limit)}>Next</button></div></div>}
    </div>}
  </section>;
}

function BenchmarkPair({ row, configuredThreshold }: { row: BenchmarkComparison; configuredThreshold: number }) {
  return <article className="comparison-pair benchmark-pair">
    <div className="comparison-images"><BenchmarkPairImage url={row.left_image_url} filename={row.left_filename} identity={row.left_identity} source={row.left_source_group} side="First"/><BenchmarkPairImage url={row.right_image_url} filename={row.right_filename} identity={row.right_identity} source={row.right_source_group} side="Second"/></div>
    <div className="comparison-score-head"><strong>{row.encoder_name}</strong><span>Cosine {row.cosine_similarity.toFixed(3)}</span></div>
    <div className="comparison-badges"><span className="chip">{row.actual_same_identity ? "Same truck" : "Different trucks"}</span><span className="chip">Configured: {row.predicted_same_configured ? "same" : "different"}</span><span className="chip">Best F1: {row.predicted_same_best_f1 ? "same" : "different"}</span>{!row.strict_eligible && <span className="chip">Excluded · {row.exclusion_reason}</span>}</div>
    <dl className="comparison-scores"><div><dt>Configured threshold</dt><dd>{configuredThreshold.toFixed(3)}</dd></div><div><dt>Exploratory best threshold</dt><dd>{row.best_f1_threshold.toFixed(3)}</dd></div><div><dt>Strict metric</dt><dd>{row.strict_eligible ? "Eligible" : "Excluded"}</dd></div><div><dt>Encoder fingerprint</dt><dd title={row.encoder_fingerprint}>{row.encoder_fingerprint.slice(0, 12)}</dd></div></dl>
  </article>;
}

function BenchmarkPairImage({ url, filename, identity, source, side }: { url: string; filename: string; identity: string; source: string; side: string }) {
  return <a className="comparison-image" href={url} target="_blank" rel="noreferrer"><span>{side} image</span><img src={url} alt={`${side} benchmark image ${filename}`}/><strong>{identity} · {filename}</strong><small>Viewpoint {source}</small></a>;
}

function DeleteExperimentDialog({ preview, busy, error, close, confirm }: { preview: DeletionPreview; busy: boolean; error: string; close: () => void; confirm: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => { dialog.current?.showModal(); }, []);
  return <dialog ref={dialog} className="reid-delete-dialog" aria-labelledby="reid-delete-title" onCancel={event => { event.preventDefault(); if (!busy) close(); }}>
    <h3 id="reid-delete-title">Delete experiment permanently?</h3>
    <p><strong>{preview.name}</strong></p>
    <p>This removes {preview.observations} vehicle observations and {preview.files} stored files, including uploaded media, crops, embeddings, and results. This cannot be undone.</p>
    <p>{preview.identities_removed.length} vehicle IDs will be removed; {preview.identities_preserved.length} shared IDs will be preserved.</p>
    {preview.experiments_to_refresh.length > 0 && <p>{preview.experiments_to_refresh.length} other experiments will need matching refreshed.</p>}
    {preview.blockers.length > 0 && <div className="notice warning"><strong>Deletion is blocked</strong><ul>{preview.blockers.map(item => <li key={item.kind + item.id}>{item.name} ({item.kind})</li>)}</ul><p>Research datasets and trained models are protected.</p></div>}
    {error && <p role="alert" className="notice error">{error}</p>}
    <div className="button-row"><button className="button secondary" disabled={busy} onClick={close}>Cancel deletion</button><button className="button danger-button" disabled={busy || preview.blockers.length > 0} onClick={confirm}>{busy ? "Deleting…" : "Delete permanently"}</button></div>
  </dialog>;
}

type ComparisonFlag = "all" | "winning" | "assigned" | "decision-reference";

function ComparisonBrowser({ experimentId, defaultEncoder, title, observationId, disabled = false }: { experimentId: string; defaultEncoder: string; title: string; observationId?: string; disabled?: boolean }) {
  const [open, setOpen] = useState(false);
  const [page, setPage] = useState<ReIDComparisonPage | null>(null);
  const [encoderId, setEncoderId] = useState(defaultEncoder);
  const [queryId, setQueryId] = useState("");
  const [vehicleId, setVehicleId] = useState("");
  const [flag, setFlag] = useState<ComparisonFlag>("all");
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");
  const limit = 24;
  useEffect(() => { setEncoderId(defaultEncoder); setOffset(0); }, [experimentId, defaultEncoder]);
  useEffect(() => {
    if (!open) return;
    let alive = true;
    setLoading(true); setLoadError("");
    void reid.comparisons(experimentId, { observation_id: observationId || queryId || undefined, encoder_id: encoderId || undefined, candidate_vehicle_id: vehicleId || undefined, decision_flag: flag, offset, limit })
      .then(value => { if (alive) setPage(value); })
      .catch(exc => { if (alive) setLoadError(String(exc instanceof Error ? exc.message : exc)); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [open, experimentId, observationId, queryId, encoderId, vehicleId, flag, offset]);
  const reset = (change: () => void) => { setOffset(0); change(); };
  const first = page?.items[0];
  return <section className={"comparison-browser " + (observationId ? "comparison-inline" : "comparison-experiment")}>
    <button className="button secondary comparison-toggle" disabled={disabled} aria-expanded={open} onClick={() => setOpen(value => !value)}>{open ? "Hide comparisons" : observationId ? "View comparisons" : "Open image comparison table"}</button>
    {open && <div className="comparison-body">
      <div className="reid-toolbar"><div><h3>{title}</h3><p className="subtle">Every eligible trusted gallery crop scored during matching. Ordered by query, encoder, and raw cosine rank.</p></div>{page && <span className="chip">{page.total} comparisons</span>}</div>
      <div className="comparison-filters">
        {!observationId && <label><span>Query crop</span><select aria-label="Comparison query" value={queryId} onChange={event => reset(() => setQueryId(event.target.value))}><option value="">All queries</option>{page?.available_observations.map(item => <option key={item.id} value={item.id}>{item.filename} · {item.id.slice(-8)}</option>)}</select></label>}
        <label><span>Encoder</span><select aria-label={observationId ? "Observation comparison encoder" : "Comparison encoder"} value={encoderId} onChange={event => reset(() => setEncoderId(event.target.value))}><option value="">All encoders</option>{page?.available_encoders.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
        <label><span>Candidate vehicle ID</span><select aria-label="Comparison vehicle ID" value={vehicleId} onChange={event => reset(() => setVehicleId(event.target.value))}><option value="">All vehicle IDs</option>{page?.available_vehicle_ids.map(id => <option key={id}>{id}</option>)}</select></label>
        <label><span>Decision role</span><select aria-label="Comparison decision role" value={flag} onChange={event => reset(() => setFlag(event.target.value as ComparisonFlag))}><option value="all">All comparisons</option><option value="winning">Winning identity</option><option value="assigned">Assigned identity</option><option value="decision-reference">Exact decision reference</option></select></label>
      </div>
      {loadError && <p role="alert" className="notice error">{loadError}</p>}
      {loading && <p className="subtle comparison-loading">Loading image comparisons…</p>}
      {!loading && page?.total === 0 && <p className="notice">No eligible gallery comparisons match these filters. A first experiment with no saved references will have no comparison pairs.</p>}
      {!loading && first && <div className={observationId ? "comparison-pair-layout fixed-query" : "comparison-pair-layout"}>
        {observationId && <ComparisonImage row={first} query/>}
        <div className="comparison-reference-list">{page?.items.map((row, index) => <ComparisonPair key={row.encoder_id + row.query_observation_id + row.query_crop_index + row.reference_observation_id + row.reference_crop_index + index} row={row} showQuery={!observationId}/>)}</div>
      </div>}
      {page && page.total > 0 && <div className="comparison-pagination"><span>{offset + 1}–{Math.min(offset + limit, page.total)} of {page.total}</span><div className="button-row"><button className="button secondary" disabled={loading || offset === 0} onClick={() => setOffset(value => Math.max(0, value - limit))}>Previous</button><button className="button secondary" disabled={loading || offset + limit >= page.total} onClick={() => setOffset(value => value + limit)}>Next</button></div></div>}
    </div>}
  </section>;
}

function ComparisonImage({ row, query }: { row: ReIDComparison; query: boolean }) {
  const url = query ? row.query_image_url : row.reference_image_url;
  const filename = query ? row.query_filename : row.reference_filename;
  const camera = query ? row.query_camera_id : row.reference_camera_id;
  const className = query ? row.query_class_name : row.reference_class_name;
  const captured = query ? row.query_capture_time : row.reference_capture_time;
  const timestamp = captured == null ? "time unknown" : new Date(typeof captured === "number" ? captured * 1000 : captured).toLocaleString();
  return <a className="comparison-image" href={url} target="_blank" rel="noreferrer"><span>{query ? "Query crop" : "Gallery reference"}</span><img src={url} alt={(query ? "Query " : "Reference ") + filename}/><strong>{filename}</strong><small>{camera} · {className} · {timestamp}</small></a>;
}

function ComparisonPair({ row, showQuery }: { row: ReIDComparison; showQuery: boolean }) {
  const score = (value: number | null | undefined) => value == null ? "—" : value.toFixed(3);
  return <article className="comparison-pair">
    <div className="comparison-images">{showQuery && <ComparisonImage row={row} query/>}<ComparisonImage row={row} query={false}/></div>
    <div className="comparison-score-head"><strong>{row.encoder_name}</strong><span>Cosine #{row.cosine_rank} · {score(row.cosine_similarity)}</span></div>
    <div className="comparison-badges">{row.is_winning_identity && <span className="chip">Winning identity</span>}{row.is_assigned_identity && <span className="chip">Assigned identity</span>}{row.is_decision_reference && <span className="chip">Decision reference</span>}</div>
    <dl className="comparison-scores"><div><dt>Candidate ID</dt><dd>{row.candidate_vehicle_id}</dd></div><div><dt>Decision</dt><dd>{statusLabel(row.decision_status)}</dd></div><div><dt>Detected / assigned</dt><dd>{row.detected_vehicle_id || "—"} / {row.assigned_vehicle_id || "—"}</dd></div><div><dt>Color / combined</dt><dd>{score(row.color_similarity)} / {score(row.combined_score)}</dd></div><div><dt>Color weight / context</dt><dd>{score(row.effective_color_weight)} / {score(row.context_score)}</dd></div><div><dt>Encoder fingerprint</dt><dd title={row.encoder_fingerprint}>{row.encoder_fingerprint.slice(0, 12)}</dd></div></dl>
  </article>;
}

function TrackCard({ track, identities, encoders, activeEncoder, comparisonReady, disabled, act, focus }: { track: ReIDTrack; identities: string[]; encoders: ReIDEncoder[]; activeEncoder: string; comparisonReady: boolean; disabled: boolean; act: (fn: () => Promise<unknown>) => Promise<void>; focus: (id: string) => void }) {
  const [label, setLabel] = useState(track.identity || "");
  const [globalId, setGlobalId] = useState(track.global_id || "");
  const [showImage, setShowImage] = useState(false);
  useEffect(() => setGlobalId(track.global_id || ""), [track.global_id]);
  return <article className="reid-track">
    <div className="reid-toolbar"><strong><Camera size={14}/> {track.camera_id} · local {track.local_id}</strong><span className="chip">{track.global_id || "Needs review"}</span></div>
    <p className="subtle">{"Crop · " + track.class_name} · {track.reviewed ? "Reviewed site identity" : "Provisional"}{track.excluded ? " · Excluded from evaluation" : ""}</p>
    <div className="reid-crops">{track.crops.map((c, index) => <a key={c.frame} href={"/api/reid/tracks/" + track.id + "/crops/" + index} target="_blank" rel="noreferrer"><img src={"/api/reid/tracks/" + track.id + "/crops/" + index} alt={"Vehicle crop " + (index + 1)}/></a>)}</div>
    <button className="button secondary" onClick={() => setShowImage(v => !v)}>{showImage ? "Hide source" : "Inspect image"}</button>
    {showImage && <img className="reid-preview" src={"/api/reid/tracks/" + track.id + "/image"} alt="Original vehicle image"/>}
    {Object.entries(track.assignments).map(([key, assignment]) => <div key={key} className="reid-match"><strong>{encoders.find(e => e.id === key)?.name || key}</strong><span>{assignment.global_id || "Review suggested matches"} · {statusLabel(assignment.status)}{assignment.similarity !== null ? (assignment.scoring_version ? " · combined " : " · cosine ") + assignment.similarity.toFixed(3) : ""}</span>{assignment.reason && <small>{assignment.reason}</small>}{assignment.provenance === "experimental" && <small>Experimental matching thresholds</small>}{assignment.candidates.map(c => <button key={c.global_id} className="button secondary" onClick={() => setGlobalId(c.global_id)}>{c.reference_track_id && <img className="reid-reference" src={"/api/reid/tracks/" + c.reference_track_id + "/crops/" + (c.reference_crop_index || 0)} alt={"Saved reference " + c.global_id}/>} {c.global_id}<ScoreDetails score={c}/></button>)}{assignment.neighbors?.map(n => <button key={n.track_id} className="button secondary" onClick={() => focus(n.track_id)}>Similar observation · combined {n.similarity.toFixed(3)}</button>)}</div>)}
    <ComparisonBrowser experimentId={track.experiment_id} observationId={track.id} defaultEncoder={activeEncoder} title="Observation image comparisons" disabled={!comparisonReady}/>
    <div className="field-row reid-spacing"><label><span>Site vehicle identity</span><select aria-label={"Site identity for " + track.id} value={globalId} onChange={e => setGlobalId(e.target.value)}><option value="">Create new vehicle ID</option>{identities.map(id => <option key={id} value={id}>{id}</option>)}</select></label><button className="button primary" disabled={disabled} onClick={() => void act(() => reid.review(track.id, globalId))}>Confirm identity</button></div>
    <details className="reid-spacing"><summary>Optional identity labels</summary><label><span>True identity label · e.g. SITE_TRUCK_A</span><input aria-label={"True identity for " + track.id} value={label} onChange={e => setLabel(e.target.value)}/></label><div className="button-row"><button className="button secondary" disabled={disabled} onClick={() => void act(() => reid.annotate(track.id, label.trim() || null, track.excluded))}>Save label</button><button className="button secondary" disabled={disabled} onClick={() => void act(() => reid.annotate(track.id, label.trim() || null, !track.excluded))}>{track.excluded ? "Include in evaluation" : "Exclude from evaluation"}</button></div></details>
  </article>;
}

function SiteEditor({ site, encoders, busy, act }: { site: ReIDSite; encoders: ReIDEncoder[]; busy: boolean; act: (fn: () => Promise<unknown>) => Promise<void> }) {
  const [cameras, setCameras] = useState(site.cameras.map(c => c.id).join("\n"));
  const [transitions, setTransitions] = useState(site.transitions);
  const [encoder, setEncoder] = useState(site.active_encoder);
  const [matching, setMatching] = useState<MatchingSettings>({ mode: "review", threshold: .85, margin: .05, new_threshold: .65, color_weight: .25, ...site.matching });
  return <div className="reid-spacing"><h3>Identity matching</h3><label><span>Matching mode</span><select aria-label="Matching mode" value={matching.mode} onChange={e => setMatching(v => ({ ...v, mode: e.target.value as MatchingSettings["mode"] }))}><option value="automatic">Automatically reuse IDs</option><option value="review">Review matches (automatic only with calibration)</option></select></label>
    <div className="field-row three">{(["threshold", "margin", "new_threshold"] as const).map(key => <label key={key}><span>{{ threshold: "Automatic match threshold", margin: "Minimum lead over next match", new_threshold: "New vehicle below similarity" }[key]}</span><input aria-label={key} type="number" min={0} max={1} step=".01" value={matching[key]} onChange={e => setMatching(v => ({ ...v, [key]: Number(e.target.value) }))}/></label>)}</div>
    <label className="reid-spacing"><span>Color weight · 0 restores appearance-only matching</span><input aria-label="Color weight" type="number" min={0} max={.5} step=".05" value={matching.color_weight} onChange={e => setMatching(v => ({ ...v, color_weight: Number(e.target.value) }))}/></label>
    <p className="subtle">Combined score = {(1 - matching.color_weight).toFixed(2)} × appearance + {matching.color_weight.toFixed(2)} × color. Defaults are experimental. Calibration applies only to its encoder, color weight, and scoring version. Refresh existing experiments to apply changes. Fine-tuning is optional.</p>
    <button className="button primary" disabled={busy} onClick={() => void act(() => reid.updateSite(site.id, { name: site.name, cameras: site.cameras, transitions: site.transitions, matching }))}>Save matching settings</button>
    <h3>{site.name} · camera settings</h3><label><span>Camera IDs · one per line</span><textarea value={cameras} onChange={e => setCameras(e.target.value)}/></label><p className="subtle">Optional travel-time rules support appearance matching only when recording times are known.</p>{transitions.map((t, index) => <div className="field-row" key={index}><input aria-label="Transition source" placeholder="From camera" value={t.source} onChange={e => setTransitions(rows => rows.map((r, i) => i === index ? { ...r, source: e.target.value } : r))}/><input aria-label="Transition destination" placeholder="To camera" value={t.destination} onChange={e => setTransitions(rows => rows.map((r, i) => i === index ? { ...r, destination: e.target.value } : r))}/><label><span>Minimum seconds</span><input type="number" value={t.min_seconds} onChange={e => setTransitions(rows => rows.map((r, i) => i === index ? { ...r, min_seconds: Number(e.target.value) } : r))}/></label><label><span>Maximum seconds</span><input type="number" value={t.max_seconds} onChange={e => setTransitions(rows => rows.map((r, i) => i === index ? { ...r, max_seconds: Number(e.target.value) } : r))}/></label><button className="button secondary" onClick={() => setTransitions(rows => rows.filter((_, i) => i !== index))}>Remove rule</button></div>)}<div className="button-row reid-spacing"><button className="button secondary" onClick={() => setTransitions(rows => [...rows, { source: "", destination: "", min_seconds: 0, max_seconds: 3600 }])}>Add travel-time rule</button><button className="button primary" disabled={busy} onClick={() => void act(() => reid.updateSite(site.id, { name: site.name, cameras: cameras.split(/\r?\n/).map(s => s.trim()).filter(Boolean).map(id => ({ id, name: id })), transitions }))}>Save cameras & rules</button></div><label className="reid-spacing"><span>Site matching encoder · currently {site.active_encoder}</span><select value={encoder} onChange={e => setEncoder(e.target.value)}>{encoders.map(e => <option key={e.id} value={e.id}>{e.name}</option>)}</select></label><button className="button secondary" disabled={busy} onClick={() => void act(() => reid.promote(site.id, encoder))}>Use encoder & rebuild site gallery</button></div>;
}

function UploadThumbnail({ file }: { file: File }) {
  const [url, setUrl] = useState("");
  useEffect(() => { const value = URL.createObjectURL(file); setUrl(value); return () => URL.revokeObjectURL(value); }, [file]);
  return url ? <img className="reid-reference" src={url} alt={"Upload preview " + file.name}/> : null;
}

function ImageResults({ job }: { job: ReIDJob }) {
  const images = Object.entries(job.artifacts).filter(([, url]) => url.endsWith(".jpg"));
  const [chosen, setChosen] = useState("");
  const url = images.some(([, value]) => value === chosen) ? chosen : images[0]?.[1];
  if (!url) return null;
  return <label className="reid-spacing"><span>Annotated image</span><select aria-label="Annotated image" value={url} onChange={e => setChosen(e.target.value)}>{images.map(([name, value]) => <option key={value} value={value}>{name}</option>)}</select><img className="reid-preview" src={url + "?v=" + encodeURIComponent((job.finished_at || "") + job.state)} alt="Annotated vehicle result"/></label>;
}

function ScoreDetails({ score }: { score: ReIDScore & { similarity: number } }) {
  if (!score.scoring_version) return <span className="reid-score-details">Appearance cosine {score.similarity.toFixed(3)} · legacy result</span>;
  return <span className="reid-score-details">
    <span>Appearance cosine {score.appearance_similarity?.toFixed(3) ?? "—"}</span>
    <span>Color similarity {score.color_similarity == null ? "unavailable · appearance only" : score.color_similarity.toFixed(3)}</span>
    <strong>Combined match score {score.similarity.toFixed(3)}</strong>
    <small>Color weight {score.effective_color_weight?.toFixed(2)}{score.color_methods ? " · " + score.color_methods.query + " / " + score.color_methods.reference : ""}</small>
  </span>;
}
