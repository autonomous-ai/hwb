import { defineWorkersConfig, readD1Migrations } from "@cloudflare/vitest-pool-workers/config";

// Tests run inside workerd (the real Workers runtime) with local R2 and D1.
export default defineWorkersConfig(async () => ({
  test: {
    setupFiles: ["./test/apply-migrations.js"],
    poolOptions: {
      workers: {
        singleWorker: true,
        wrangler: { configPath: "./wrangler.toml" },
        miniflare: {
          bindings: {
            TEST_MIGRATIONS: await readD1Migrations("./migrations"),
            UPLOAD_TOKEN: "test-upload-token",
            ADMIN_TOKEN: "test-admin-token",
            PUBLIC_BASE_URL: "https://hwb.example.com",
          },
        },
      },
    },
  },
}));
