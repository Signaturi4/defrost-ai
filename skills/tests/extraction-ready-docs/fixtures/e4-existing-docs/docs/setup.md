# Setup

This is how you set it up. Install Node 20 and pnpm, then run pnpm install in the repo root. Copy .env.example to .env and set the DATABASE_URL to your local Postgres. Run pnpm dev to start the app on port 3000, it will reload when you change files and it also starts the worker that sends invoice emails in the background so you don't need to start that separately.
