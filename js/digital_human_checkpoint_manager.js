import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ASSEMBLER = "MiniMaxH3StreamCheckpointDigitalHumanToVHS";
const GATE = "MiniMaxH3CheckpointResume";
const PLANNER = "MiniMaxH3AudioTimelinePlan";
const VIDEO_PREVIEW = "VHS_VideoCombine";
const INDEX_CONTROL = "h3_digital_human_current_clip";
const API_BASE = "/h3_digital_human/checkpoints";

let session = null;
let managedSubmission = false;
let projectPanel = null;

function nodeType(node) {
    return node?.comfyClass ?? node?.type ?? "";
}

function widget(node, name) {
    return node?.widgets?.find((item) => item.name === name);
}

function setStatus(node, message) {
    node.__h3CheckpointStatus = String(message);
    const button = widget(node, "Checkpoint status");
    if (button) button.name = `Checkpoint status: ${message}`;
    if (projectPanel?.assembler === node && projectPanel.status) {
        projectPanel.status.textContent = String(message);
    }
    node.setDirtyCanvas?.(true, true);
}

function sleep(milliseconds) {
    return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function responseJson(response) {
    let body = null;
    try { body = await response.json(); } catch (_) {}
    if (!response.ok) {
        throw new Error(body?.error || `HTTP ${response.status}`);
    }
    return body;
}

async function queueIsEmpty() {
    const response = await api.fetchApi("/api/queue");
    const body = await responseJson(response);
    return !(body?.queue_running?.length || body?.queue_pending?.length);
}

function graphNodes() {
    return app.graph?._nodes ?? app.graph?.nodes ?? [];
}

function installNormalQueueRedirect() {
    if (app.__h3CheckpointQueueRedirectInstalled) return;
    app.__h3CheckpointQueueRedirectInstalled = true;
    const originalQueuePrompt = app.queuePrompt.bind(app);
    app.queuePrompt = async function (number = 0, batchCount = 1, selectedNodeIds) {
        const isPartial = Array.isArray(selectedNodeIds) && selectedNodeIds.length > 0;
        const gates = graphNodes().filter((node) => nodeType(node) === GATE);
        const assemblers = graphNodes().filter((node) => nodeType(node) === ASSEMBLER);
        if (!managedSubmission && !isPartial && gates.length && assemblers.length) {
            if (assemblers.length !== 1) {
                for (const assembler of assemblers) {
                    setStatus(assembler, "multiple checkpoint assemblers found; use the intended Run/resume button");
                }
                return false;
            }
            const assembler = assemblers[0];
            if (session) {
                setStatus(assembler, "managed run already active; normal Queue ignored");
                return false;
            }
            setStatus(
                assembler,
                Number(batchCount) > 1
                    ? "normal Queue redirected to one managed run; batch count ignored"
                    : "normal Queue redirected to safe managed run"
            );
            await startManagedRun(assembler);
            return true;
        }
        return originalQueuePrompt(number, batchCount, selectedNodeIds);
    };
}

function switchProjectPrefix(assembler, prefix) {
    const oldPrefix = String(widget(assembler, "checkpoint_prefix")?.value ?? "").trim();
    const targets = graphNodes().filter((node) => {
        const value = widget(node, "checkpoint_prefix");
        return value && String(value.value ?? "").trim() === oldPrefix;
    });
    for (const node of targets) {
        const value = widget(node, "checkpoint_prefix");
        if (!value) continue;
        value.value = prefix;
        value.callback?.(prefix);
        node.setDirtyCanvas?.(true, true);
    }
    app.graph?.setDirtyCanvas?.(true, true);
}

function runConfiguration(assembler, options = {}) {
    const prefix = String(widget(assembler, "checkpoint_prefix")?.value ?? "").trim();
    if (!prefix) throw new Error("checkpoint_prefix is empty");
    const planners = graphNodes().filter((node) => nodeType(node) === PLANNER);
    const gates = graphNodes().filter((node) => nodeType(node) === GATE);
    const indices = graphNodes().filter((node) =>
        node?.properties?.h3_digital_human_role === INDEX_CONTROL
    );
    if (planners.length !== 1) throw new Error(`expected one audio timeline planner, found ${planners.length}`);
    if (gates.length !== 1) throw new Error(`expected one reusable checkpoint gate, found ${gates.length}`);
    if (indices.length !== 1) throw new Error(`expected one managed clip-index control, found ${indices.length}`);
    const gatePrefix = String(widget(gates[0], "checkpoint_prefix")?.value ?? "").trim();
    if (gatePrefix !== prefix) throw new Error("reusable checkpoint gate prefix differs from assembler");
    if (gates[0].mode === 4) throw new Error("reusable checkpoint gate is bypassed");
    const startIndex = Math.max(1, Math.trunc(Number(options.startIndex ?? 1)));
    return {
        assembler,
        prefix,
        planner: planners[0],
        gate: gates[0],
        indexControl: indices[0],
        count: 0,
        index: startIndex,
        startIndex,
        retries: Math.max(0, Math.trunc(Number(
            widget(assembler, "generation_retries")?.value ?? 1
        ))),
        delay: Math.max(0, Math.trunc(Number(
            widget(assembler, "cleanup_delay_ms")?.value ?? 1500
        ))),
        attempt: 0,
        phase: "plan",
        planResult: null,
        stopped: false,
        awaiting: false,
        pauseAfterCurrent: false,
        forceIndex: options.forceIndex ?? null,
    };
}

async function releaseResources() {
    const response = await api.fetchApi(`${API_BASE}/cleanup`, { method: "POST" });
    await responseJson(response);
}

async function submitCurrent() {
    const current = session;
    if (!current || current.stopped || current.awaiting) return;
    if (!(await queueIsEmpty())) {
        throw new Error("ComfyUI queue is not empty; wait or clear it before starting managed mode");
    }
    let target = current.assembler;
    if (current.phase === "plan") {
        target = current.planner;
        const indexWidget = widget(current.indexControl, "value") ?? current.indexControl?.widgets?.[0];
        if (indexWidget) {
            indexWidget.value = 1;
            indexWidget.callback?.(1);
        }
    }
    if (current.phase === "preview") target = current.preview;
    if (current.phase === "clips") {
        target = current.gate;
        const indexWidget = widget(current.indexControl, "value") ?? current.indexControl?.widgets?.[0];
        if (!indexWidget) throw new Error("managed clip-index widget is unavailable");
        indexWidget.value = current.index;
        indexWidget.callback?.(current.index);
        current.indexControl.setDirtyCanvas?.(true, true);
        const reuse = widget(current.gate, "reuse_checkpoint");
        const shouldReuse = Number(current.forceIndex) !== Number(current.index);
        if (reuse) {
            reuse.value = shouldReuse;
            reuse.callback?.(shouldReuse);
        }
    }
    const label = current.phase === "plan" ? "audio timeline plan"
        : current.phase === "preview" ? `Clip ${current.index} preview`
        : current.phase === "assemble" ? "final assembly"
        : `Clip ${current.index}/${current.count}`;
    setStatus(current.assembler, `queueing ${label}`);
    current.awaiting = true;
    try {
        current.prepareSubmit?.();
        managedSubmission = true;
        const accepted = await app.queuePrompt(0, 1, [String(target.id)]);
        if (accepted === false) throw new Error("ComfyUI rejected the prompt");
    } catch (error) {
        current.awaiting = false;
        throw error;
    } finally {
        managedSubmission = false;
        current.restoreAfterSubmit?.();
        current.restoreAfterSubmit = null;
    }
}

async function advanceAfterSuccess() {
    const current = session;
    if (!current || current.stopped || !current.awaiting) return;
    current.awaiting = false;
    if (current.phase === "assemble") {
        setStatus(current.assembler, "complete");
        session = null;
        return;
    }
    if (current.phase === "preview") {
        setStatus(current.assembler, `Clip ${current.index} preview complete`);
        session = null;
        return;
    }
    if (current.phase === "plan") {
        const plan = current.planResult;
        const count = Math.trunc(Number(plan?.clip_count ?? 0));
        if (!(count >= 1 && count <= 20)) {
            throw new Error("audio timeline planner did not return a valid clip count");
        }
        if (current.startIndex > count) {
            throw new Error(`requested Clip ${current.startIndex}, but audio requires only ${count} clips`);
        }
        current.count = count;
        current.index = current.startIndex;
        current.assembler.__h3PlannedCount = count;
        current.assembler.__h3TimelinePlan = plan;
        current.phase = "clips";
        current.attempt = 0;
        setStatus(
            current.assembler,
            `audio ${Number(plan.duration_seconds).toFixed(3)}s → ${count} clips; starting Clip ${current.index}`
        );
        // Let ComfyUI remove the completed planner prompt from /api/queue
        // before submitting the first independently checkpointed Clip.
        await sleep(250);
        if (session !== current || current.stopped) return;
        return submitCurrent();
    }
    const completed = current.index;
    if (Number(current.forceIndex) === completed) {
        const reuse = widget(current.gate, "reuse_checkpoint");
        if (reuse) {
            reuse.value = true;
            reuse.callback?.(true);
        }
        current.forceIndex = null;
    }
    setStatus(current.assembler, `Clip ${completed} checkpointed; releasing models`);
    await releaseResources();
    if (current.pauseAfterCurrent) {
        current.stopped = true;
        setStatus(current.assembler, `paused after Clip ${completed}; checkpoints kept`);
        session = null;
        return;
    }
    await sleep(current.delay);
    if (session !== current || current.stopped) return;
    current.index += 1;
    current.attempt = 0;
    if (current.index > current.count) current.phase = "assemble";
    await submitCurrent();
}

async function handleFailure(kind) {
    const current = session;
    if (!current || current.stopped || !current.awaiting) return;
    current.awaiting = false;
    if (kind === "interrupted") {
        current.stopped = true;
        try { await releaseResources(); } catch (_) {}
        setStatus(current.assembler, "paused after interruption; checkpoints kept");
        session = null;
        return;
    }
    if (current.attempt < current.retries) {
        current.attempt += 1;
        setStatus(current.assembler, `retry ${current.attempt}/${current.retries}`);
        try { await releaseResources(); } catch (_) {}
        await sleep(current.delay);
        if (session === current && !current.stopped) await submitCurrent();
        return;
    }
    current.stopped = true;
    try { await releaseResources(); } catch (_) {}
    setStatus(current.assembler, "paused after error; press Run/resume to continue");
    session = null;
}

async function startManagedRun(assembler, options = {}) {
    let current = null;
    try {
        if (session) throw new Error("a managed checkpoint run is already active");
        current = runConfiguration(assembler, options);
        session = current;
        setStatus(assembler, "measuring master audio");
        await submitCurrent();
    } catch (error) {
        if (session === current) session = null;
        setStatus(assembler, `not started: ${error.message || error}`);
    }
}

async function startAssembly(assembler, preview = false) {
    if (session) {
        setStatus(assembler, "assembly not started: a managed run is already active");
        return;
    }
    const save = widget(assembler, "save_output");
    const crf = widget(assembler, "crf");
    const oldSave = save?.value;
    const oldCrf = crf?.value;
    const current = {
        assembler, phase: "assemble", stopped: false, awaiting: false,
        attempt: 0,
        retries: Math.max(0, Math.trunc(Number(
            widget(assembler, "generation_retries")?.value ?? 1
        ))),
        delay: Math.max(0, Math.trunc(Number(
            widget(assembler, "cleanup_delay_ms")?.value ?? 1500
        ))),
        prepareSubmit() {
            if (save) { save.value = !preview; save.callback?.(!preview); }
            if (preview && crf) {
                crf.value = Math.max(26, Number(oldCrf ?? 19));
                crf.callback?.(crf.value);
            }
            current.restoreAfterSubmit = () => {
                if (save) { save.value = oldSave; save.callback?.(oldSave); }
                if (crf) { crf.value = oldCrf; crf.callback?.(oldCrf); }
            };
        },
    };
    session = current;
    setStatus(assembler, preview ? "queueing preview assembly" : "queueing final export");
    try {
        await submitCurrent();
    } catch (error) {
        current.restoreAfterSubmit?.();
        session = null;
        setStatus(assembler, `assembly not started: ${error.message || error}`);
    }
}

async function startClipPreview(assembler, index) {
    if (session) {
        setStatus(assembler, "preview not started: a managed run is already active");
        return;
    }
    const gates = graphNodes().filter((node) => nodeType(node) === GATE);
    const previews = graphNodes().filter((node) =>
        nodeType(node) === VIDEO_PREVIEW
        && node?.properties?.h3_digital_human_role === "current_clip_preview"
    );
    const indices = graphNodes().filter((node) =>
        node?.properties?.h3_digital_human_role === INDEX_CONTROL
    );
    if (gates.length !== 1 || previews.length !== 1 || indices.length !== 1) {
        setStatus(assembler, "preview not started: reusable preview graph is incomplete");
        return;
    }
    const indexWidget = widget(indices[0], "value") ?? indices[0]?.widgets?.[0];
    if (!indexWidget) {
        setStatus(assembler, "preview not started: clip-index widget is unavailable");
        return;
    }
    indexWidget.value = Number(index);
    indexWidget.callback?.(Number(index));
    const reuse = widget(gates[0], "reuse_checkpoint");
    if (reuse) {
        reuse.value = true;
        reuse.callback?.(true);
    }
    const current = {
        assembler, preview: previews[0], phase: "preview", index: Number(index),
        stopped: false, awaiting: false, attempt: 0, retries: 0, delay: 0,
    };
    session = current;
    try {
        await submitCurrent();
    } catch (error) {
        session = null;
        setStatus(assembler, `preview not started: ${error.message || error}`);
    }
}

function pauseManagedRun(assembler) {
    if (!session || session.assembler !== assembler) {
        setStatus(assembler, "no managed run is active");
        return;
    }
    session.pauseAfterCurrent = true;
    setStatus(assembler, "pause requested; finishing current prompt");
}

function stopManagedRun(assembler) {
    if (!session) {
        setStatus(assembler, "idle");
        return;
    }
    session.pauseAfterCurrent = true;
    setStatus(assembler, "stop requested; no new prompt after the current one");
}

function element(tag, attributes = {}, ...children) {
    const value = document.createElement(tag);
    for (const [key, item] of Object.entries(attributes)) {
        if (key === "class") value.className = item;
        else if (key === "text") value.textContent = item;
        else if (key.startsWith("on")) value.addEventListener(key.slice(2), item);
        else value.setAttribute(key, item);
    }
    value.append(...children);
    return value;
}

function bytes(value) {
    if (value == null) return "unknown";
    let amount = Number(value ?? 0);
    const units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let unit = 0;
    while (amount >= 1024 && unit < units.length - 1) {
        amount /= 1024;
        unit += 1;
    }
    return `${amount.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function timestamp(nanoseconds) {
    const value = Number(nanoseconds ?? 0) / 1e6;
    return value > 0 ? new Date(value).toLocaleString() : "—";
}

async function showProjectManager(assembler) {
    const prefix = String(widget(assembler, "checkpoint_prefix")?.value ?? "").trim();
    const [status, projectData] = await Promise.all([
        api.fetchApi(`${API_BASE}?prefix=${encodeURIComponent(prefix)}`).then(responseJson),
        api.fetchApi(`${API_BASE}/projects`).then(responseJson),
    ]);
    const clipMap = new Map((status.clips ?? []).map((clip) => [clip.index, clip]));
    const manifestCount = Math.max(0, ...(status.clips ?? []).map((clip) => Number(clip.index) || 0));
    const planner = graphNodes().find((node) => nodeType(node) === PLANNER);
    const fallbackCount = Number(widget(planner, "manual_clips")?.value ?? 1);
    const count = Math.max(1, Math.trunc(Number(
        assembler.__h3PlannedCount ?? (manifestCount || fallbackCount)
    )));
    const ready = Array.from({ length: count }, (_, offset) =>
        clipMap.get(offset + 1)?.ready === true).filter(Boolean).length;
    const shade = element("div", { class: "h3mr-shade" });
    const panel = element("div", { class: "h3mr-panel" });
    const closePanel = () => {
        if (projectPanel?.shade === shade) projectPanel = null;
        shade.remove();
    };
    const runtimeStatus = element("strong", {
        class: "h3mr-runtime",
        text: assembler.__h3CheckpointStatus ?? "idle",
    });
    projectPanel = { assembler, status: runtimeStatus, shade };

    const projects = element("select", { class: "h3mr-projects" });
    const known = new Set();
    for (const project of projectData.projects ?? []) {
        known.add(project.prefix);
        projects.append(element("option", {
            value: project.prefix,
            text: `${project.prefix} · ${project.ready}/${project.clips} ready`,
        }));
    }
    if (!known.has(prefix)) {
        projects.prepend(element("option", { value: prefix, text: `${prefix} · current` }));
    }
    projects.value = prefix;
    const useProject = element("button", {
        text: "Open selected",
        onclick: () => {
            switchProjectPrefix(assembler, projects.value);
            closePanel();
            void showProjectManager(assembler).catch((error) =>
                setStatus(assembler, `project error: ${error.message || error}`));
        },
    });
    const close = element("button", { text: "Close", onclick: closePanel });
    const header = element("div", { class: "h3mr-header" },
        element("div", {}, element("h2", { text: "H3 Digital Human Project" }),
            element("code", { text: prefix })),
        element("div", { class: "h3mr-project-picker" }, projects, useProject, close));

    const storage = status.storage ?? {};
    const timeline = assembler.__h3TimelinePlan;
    const metrics = element("div", { class: "h3mr-metrics" },
        element("div", {}, element("b", { text: `${ready}/${count}` }), element("span", { text: " clips ready" })),
        element("div", {}, element("b", { text: timeline ? `${Number(timeline.duration_seconds).toFixed(3)} s` : "auto on run" }), element("span", { text: " master audio" })),
        element("div", {}, element("b", { text: bytes(storage.active_bytes) }), element("span", { text: " active checkpoints" })),
        element("div", {}, element("b", { text: bytes(storage.take_bytes) }), element("span", { text: " retained takes" })),
        element("div", {}, element("b", { text: bytes(storage.free_bytes) }), element("span", { text: " disk free" })),
        element("div", {}, element("b", { text: bytes(storage.available_memory_bytes) }), element("span", { text: " RAM available" })),
        element("div", {}, element("b", { text: timestamp(status.updated_ns) }), element("span", { text: " last update" })),
    );
    const progress = element("div", { class: "h3mr-progress" },
        element("span", { style: `width:${Math.round(100 * ready / count)}%` }));
    const run = element("button", {
        class: "h3mr-primary", text: "Run / Resume / Retry",
        onclick: () => void startManagedRun(assembler),
    });
    const pause = element("button", {
        text: "Pause after current clip", onclick: () => pauseManagedRun(assembler),
    });
    const stop = element("button", {
        text: "Stop scheduling", onclick: () => stopManagedRun(assembler),
    });
    const preview = element("button", {
        text: "Assemble preview", onclick: () => void startAssembly(assembler, true),
    });
    const final = element("button", {
        text: "Final export", onclick: () => void startAssembly(assembler, false),
    });
    const refresh = element("button", {
        text: "Refresh", onclick: () => {
            closePanel();
            void showProjectManager(assembler).catch((error) =>
                setStatus(assembler, `project error: ${error.message || error}`));
        },
    });
    const toolbar = element("div", { class: "h3mr-toolbar" },
        run, pause, stop, preview, final, refresh);
    panel.append(header, metrics, progress,
        element("div", { class: "h3mr-status" },
            element("span", { text: "Run status" }), runtimeStatus), toolbar,
        element("div", { class: "h3mr-list-head" },
            element("b", { text: "Timeline" }),
            element("span", { text: "Each clip is a separate recoverable top-level job" })));

    for (let index = 1; index <= count; index += 1) {
        const clip = clipMap.get(index);
        const row = element("div", { class: "h3mr-row" });
        const state = !clip ? "missing" : clip.ready
            ? (clip.frozen_branch ? "ready · frozen branch" : "ready")
            : String(clip.reason || `stale after ${clip.stale_after_clip ?? "?"}`);
        const badgeClass = !clip ? "missing" : clip.ready ? "ready" : "stale";
        const title = element("div", { class: "h3mr-clip-title" },
            element("b", { text: `Clip ${index}` }),
            element("span", { class: `h3mr-badge ${badgeClass}`, text: state }));
        const details = element("div", { class: "h3mr-detail" },
            element("code", { text: clip ? String(clip.sha256).slice(0, 12) : "—" }),
            element("span", { text: clip ? timestamp(clip.published_ns) : "not generated" }));
        const actions = element("div", { class: "h3mr-actions" });
        const regenerate = element("button", {
            text: clip ? "Regenerate from here" : "Generate from here",
            onclick: () => void startManagedRun(assembler, {
                startIndex: index, forceIndex: index,
            }),
        });
        const inspect = element("button", {
            text: "Preview clip",
            onclick: () => void startClipPreview(assembler, index),
        });
        actions.append(regenerate);
        if (clip?.ready) actions.append(inspect);
        if (!clip) {
            row.append(title, details, actions);
            panel.append(row);
            continue;
        }
        const choices = [{ sha256: "", label: "active take" },
            ...(clip.takes ?? []).map((take, number) => ({
                sha256: take.sha256,
                label: `take ${number + 1} · ${String(take.sha256).slice(0, 10)} · ${bytes(take.size)}`,
            }))];
        const select = element("select");
        for (const choice of choices) {
            select.append(element("option", { value: choice.sha256, text: choice.label }));
        }
        const branch = element("button", {
            text: "Branch here…",
            onclick: async () => {
                const input = element("input", {
                    placeholder: "h3_digital_human/checkpoints/my_branch",
                    value: `${prefix}_branch_${clip.index}`,
                });
                const create = element("button", {
                    text: "Create branch",
                    onclick: async () => {
                        try {
                            const answer = await api.fetchApi(`${API_BASE}/branch`, {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({
                                    prefix,
                                    new_prefix: input.value,
                                    through_clip: index,
                                    take_sha256: select.value,
                                }),
                            });
                            await responseJson(answer);
                            switchProjectPrefix(assembler, input.value);
                            setStatus(assembler, `branch created: ${input.value}`);
                            closePanel();
                        } catch (error) {
                            create.textContent = `Error: ${error.message || error}`;
                        }
                    },
                });
                actions.append(input, create);
            },
        });
        actions.append(select, branch);
        row.append(title, details, actions);
        panel.append(row);
    }
    shade.append(panel);
    shade.addEventListener("click", (event) => {
        if (event.target === shade) closePanel();
    });
    document.body.append(shade);
}

app.registerExtension({
    name: "h3_digital_human.checkpoint_manager",
    setup() {
        installNormalQueueRedirect();
        api.addEventListener("executed", (event) => {
            const current = session;
            if (!current || current.phase !== "plan" || !current.awaiting) return;
            const detail = event?.detail ?? {};
            if (String(detail.node ?? "") !== String(current.planner?.id ?? "")) return;
            const timeline = detail.output?.timeline;
            if (Array.isArray(timeline) && timeline[0]) current.planResult = timeline[0];
        });
        api.addEventListener("execution_success", () => {
            void advanceAfterSuccess().catch((error) => {
                if (session) setStatus(session.assembler, `paused: ${error.message || error}`);
                session = null;
            });
        });
        api.addEventListener("execution_error", () => void handleFailure("error"));
        api.addEventListener("execution_interrupted", () => void handleFailure("interrupted"));
    },
    beforeRegisterNodeDef(nodeTypeDefinition, nodeData) {
        if (nodeData?.name !== ASSEMBLER) return;
        const original = nodeTypeDefinition.prototype.onNodeCreated;
        nodeTypeDefinition.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            const guard = widget(this, "checkpoint_prefix");
            if (guard && !guard.__h3ManagedQueueGuard) {
                guard.__h3ManagedQueueGuard = true;
                const beforeQueued = guard.beforeQueued;
                guard.beforeQueued = (...args) => {
                    const hasGates = graphNodes().some((node) => nodeType(node) === GATE);
                    if (hasGates && !managedSubmission) {
                        throw new Error(
                            "The safe Queue redirect was bypassed. Use 'Run/resume " +
                            "checkpoints → assemble' or reload the ComfyUI page so the " +
                            "checkpoint project extension can intercept normal Queue."
                        );
                    }
                    return beforeQueued?.apply(guard, args);
                };
            }
            this.addWidget("button", "Run/resume checkpoints → assemble", null,
                () => void startManagedRun(this));
            this.addWidget("button", "Open Digital Human Project…", null,
                () => void showProjectManager(this).catch((error) =>
                    setStatus(this, `project error: ${error.message || error}`)));
            this.addWidget("button", "Checkpoint status", null, () => {});
            setStatus(this, "idle");
        };
    },
});

const style = document.createElement("style");
style.textContent = `
.h3mr-shade{position:fixed;inset:0;z-index:10020;background:#000a;display:flex;align-items:center;justify-content:center}
.h3mr-panel{width:min(1040px,94vw);max-height:88vh;overflow:auto;background:#20242b;color:#eee;border:1px solid #596273;border-radius:12px;padding:20px;box-shadow:0 12px 45px #000;font:13px system-ui,sans-serif}
.h3mr-panel h2{margin:0 0 5px;font-size:22px}.h3mr-panel code{color:#a9c7ff}.h3mr-header{display:flex;justify-content:space-between;gap:16px;align-items:start}.h3mr-project-picker,.h3mr-toolbar,.h3mr-actions{display:flex;gap:7px;align-items:center;flex-wrap:wrap}.h3mr-projects{max-width:340px}
.h3mr-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px;margin:16px 0 9px}.h3mr-metrics>div{background:#292f39;border:1px solid #3b4452;border-radius:7px;padding:9px}.h3mr-metrics b,.h3mr-metrics span{display:block}.h3mr-metrics span{color:#9ca7b8;font-size:11px;margin-top:3px}.h3mr-progress{height:7px;background:#11151b;border-radius:8px;overflow:hidden;margin-bottom:14px}.h3mr-progress span{display:block;height:100%;background:#55b985}
.h3mr-toolbar{margin:10px 0 16px}.h3mr-primary{background:#2e7658!important;border-color:#62c598!important}.h3mr-status{display:flex;gap:10px;align-items:center;background:#171b21;border-radius:6px;padding:8px 10px}.h3mr-status span{color:#929dad}.h3mr-runtime{color:#b7e4cd}.h3mr-list-head{display:flex;justify-content:space-between;color:#9ca7b8;padding:6px 0}
.h3mr-row{display:grid;grid-template-columns:minmax(170px,.8fr) minmax(190px,1fr) minmax(330px,1.5fr);gap:10px;align-items:center;border-top:1px solid #3b424d;padding:11px 0}.h3mr-clip-title{display:flex;gap:8px;align-items:center}.h3mr-detail{display:flex;flex-direction:column;gap:3px;color:#9ca7b8}.h3mr-badge{font-size:10px;text-transform:uppercase;padding:3px 6px;border-radius:9px}.h3mr-badge.ready{background:#234f3c;color:#9ee3bd}.h3mr-badge.missing{background:#4a3d24;color:#efd08a}.h3mr-badge.stale{background:#562e35;color:#f0a4ae}
.h3mr-row input{max-width:250px}.h3mr-row input,.h3mr-row select,.h3mr-row button,.h3mr-panel>button,.h3mr-panel select,.h3mr-panel button{background:#303743;color:#eee;border:1px solid #667085;border-radius:5px;padding:6px}
@media(max-width:780px){.h3mr-header{display:block}.h3mr-metrics{grid-template-columns:repeat(2,1fr)}.h3mr-row{grid-template-columns:1fr}.h3mr-project-picker{margin-top:10px}}
`;
document.head.append(style);
