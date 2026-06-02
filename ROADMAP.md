# TrueNorth — Product Roadmap

> What's actively in development. These ship next.

## Status Legend
* **Research & Scoping:** Technical requirements gathering and architecture design.
* **Not Started:** Planned, specifications finalized.
* **In Progress:** Active development and coding.
* **Code Review:** Feature complete; undergoing security and implementation peer review.
* **Testing:** In quality assurance or restricted private beta.
* **Staging:** Deployed to pre-production environments for internal validation.
* **Deployed:** Production-ready and fully available in the current live release.

---

## Overview and Goals

We are evolving TrueNorth from an LLM-specific middleware into a complete infrastructure layer for modern apps. We looked closely at the biggest friction points developers face in production and built Version 2 to solve them directly.

Currently, frontend apps are still vulnerable to leaking API keys, unpredictable token usage is inflating infrastructure costs, and cloud-heavy apps break completely when network connectivity drops. V2 addresses these issues fundamentally, while also adding voice capabilities and persistent memory to make applications more accessible and human.

---

## Core Features

### 1. Secure Vault & Dynamic Configuration

**Zero Client-Side Exposure (TrueNorth Vault)**  
*Status: Testing*  
We are moving all third-party API keys—not just LLMs, but also Stripe, AWS, Twilio, etc.—into a server-side encrypted vault. Your frontend code will never touch a raw key again. Instead, clients authenticate using a single, tenant-specific `X-TrueNorth-Key`. This allows you to rotate provider keys on the backend without ever forcing users to download an app update. It works natively with React, Next.js, Expo, Go, Rust, or any standard HTTP client.

**Runtime Environment Variable Injection**  
*Status: In Progress*  
You can now call `tn.config.get("MY_SETTING")` directly from the frontend. TrueNorth serves these settings dynamically at runtime, allowing you to flip feature flags or change production configurations instantly without triggering a new build. It includes secure local device caching, so it works out-of-the-box for Expo and React Native apps even when offline.

### 2. True Offline-First Architecture

**Offline Mode for Mobile (Expo / React Native)**  
*Status: In Progress*  
Cloud dependencies shouldn't break your app when a user loses cell service. V2 introduces local session collection backed by an encrypted SQLite database on the device. If connectivity drops (crucial for use cases like rural areas with patchy signals), the data is safely queued and automatically synced when the connection returns. We also run field extraction locally via on-device models like Gemini Nano, ensuring the app remains intelligent entirely offline.

### 3. Cost & Context Optimization

**Token Optimization Engine**  
*Status: Testing*  
Long conversations usually lead to exponential token costs. After a set threshold (e.g., 10 turns), TrueNorth automatically summarizes the chat history into a dense, structured fact sheet. The LLM reads this summary instead of the raw historical text. This cuts token consumption by up to 10x per turn, reducing costs by roughly 80% for long sessions without losing conversational context.

**Budgets and Context Window Management**  
*Status: Deployed*  
You can now set strict token budgets per turn. TrueNorth algorithmically trims the context to fit your budget—dropping the oldest turns first while pinning critical extracted metadata. Additionally, the system automatically detects model-specific token limits and compresses the session history before the window fills up, completely preventing "context exceeded" errors.

### 4. Multimodal & Accessible Interfaces

**Native Voice Input & Text-to-Speech**  
*Status: In Progress*  
Text interfaces exclude a lot of users—whether they are elderly, have literacy barriers, or just have their hands full in the kitchen or behind the wheel. We are integrating the Web Speech API for browsers and `expo-speech-recognition` for mobile to allow hands-free voice input. On the output side, agents can now speak their responses using premium synthesis engines like ElevenLabs, Google TTS, or OpenAI TTS.

### 5. Persistent Personas

**Persona Marketplace & YAML Configuration**  
*Status: Not Started*  
We are decoupling conversational behavior from hardcoded prompts. Developers can now define agents with specific voices, regional dialects, and operational boundaries using standard YAML files (e.g., "Dr. Meera" for clinical intake). This makes it easy for the community to share and version-control distinct personas.

**Character AI Mode (Long-Term Memory)**  
*Status: Not Started*  
Not every interaction is a one-off task. We are adding a persistent companion mode designed to retain user context and behavioral patterns across multiple, separate sessions. Backed by a vector memory layer, this enables use cases like daily journaling partners or language tutors that actually remember what the user struggled with last week.