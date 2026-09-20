import { app } from "../../scripts/app.js";

// H3 timing, audio slicing, continuation, previews, and final export all use 24 fps.
// Lock every tagged clock widget so a UI edit cannot create a mixed-timebase run.
const CLOCKS = new Set(["fps", "frame_rate", "source_fps", "start_fps", "end_fps", "force_rate"]);
function lockClocks(node) {
    if (!node?.properties?.h3_fixed_fps) return;
    for (const widget of node.widgets ?? []) {
        if (!CLOCKS.has(widget.name) || widget._h3FixedClock) continue;
        widget._h3FixedClock = true;
        widget.value = 24;
        widget.options ??= {};
        widget.options.min = 24;
        widget.options.max = 24;
        widget.disabled = true;
        const original = widget.beforeQueued;
        widget.beforeQueued = function (...args) {
            const result = original?.apply(this, args);
            this.value = 24;
            return result;
        };
    }
}
app.registerExtension({
    name: "h3_digital_human.fixed_fps",
    nodeCreated(node) { queueMicrotask(() => lockClocks(node)); },
    afterConfigureGraph() {
        for (const node of app.graph?._nodes ?? []) lockClocks(node);
    },
});
