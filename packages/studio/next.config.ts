import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  env: {
    TRUENORTH_API_URL: process.env.TRUENORTH_API_URL ?? "http://localhost:8000",
  },
};

export default nextConfig;
