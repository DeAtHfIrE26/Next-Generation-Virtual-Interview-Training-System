import next from "eslint-config-next";

const config = [
  ...next,
  { ignores: [".next/**", "public/vision/**", "playwright-report/**", "test-results/**", "next-env.d.ts"] },
];

export default config;
