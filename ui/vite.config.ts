/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

// Builds ONE self-contained index.html (JS + CSS inlined) straight into the Python package,
// where hwb fills in the report data at run time. The rig never needs Node.
export default defineConfig({
  plugins: [react(), viteSingleFile()],
  build: {
    outDir: "../src/hwb/assets",
    emptyOutDir: true,
    target: "es2020",
    reportCompressedSize: false,
  },
  test: {
    environment: "jsdom",
    globals: true,
  },
});
