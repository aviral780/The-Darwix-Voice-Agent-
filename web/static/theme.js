// Theme switch, shared by both consoles.
//
// The preference is stored so a reviewer who prefers light does not have to
// reset it on every page, and the initial value is applied before first paint
// by the inline call at the bottom of this file to avoid a flash of the wrong
// theme.

(function () {
  const KEY = "voice-console-theme";

  function apply(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    const dark = document.getElementById("themeDark");
    const light = document.getElementById("themeLight");
    if (dark) dark.setAttribute("aria-pressed", String(theme === "dark"));
    if (light) light.setAttribute("aria-pressed", String(theme === "light"));
    try { localStorage.setItem(KEY, theme); } catch (e) { /* private mode */ }
  }

  function stored() {
    try { return localStorage.getItem(KEY); } catch (e) { return null; }
  }

  // Respect the OS preference the first time, then the explicit choice.
  const initial = stored() ||
    (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  document.documentElement.setAttribute("data-theme", initial);

  document.addEventListener("DOMContentLoaded", () => {
    apply(initial);
    const dark = document.getElementById("themeDark");
    const light = document.getElementById("themeLight");
    if (dark) dark.addEventListener("click", () => apply("dark"));
    if (light) light.addEventListener("click", () => apply("light"));
  });
})();
