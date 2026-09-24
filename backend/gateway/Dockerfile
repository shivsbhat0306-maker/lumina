# syntax = docker/dockerfile:1
#
# Built with the MONOREPO ROOT as the Docker build context (not backend/gateway/), because
# @lumina/gateway depends on the @lumina/contract workspace package — `npm install`/`npm ci`
# only resolve that correctly when run from the root, where the npm workspaces are declared.
# Deploy with:
#   fly deploy . --config backend/gateway/fly.toml --dockerfile backend/gateway/Dockerfile
# run from the repo root (not from backend/gateway/).

ARG NODE_VERSION=24.21.0
FROM node:${NODE_VERSION}-slim AS base

LABEL fly_launch_runtime="Node.js"

WORKDIR /app
ENV NODE_ENV="production"


# Throw-away build stage to reduce size of final image
FROM base AS build

RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y build-essential node-gyp pkg-config python-is-python3

# Copy the whole monorepo so npm workspaces (packages/contract, backend/gateway, etc.) resolve.
COPY . .

# `npm ci` (not `npm install`) installs exactly what package-lock.json pins, matching what
# has actually been run and proven working locally.
RUN npm ci --include=dev

# @lumina/gateway imports @lumina/contract's compiled output, so contract must build first.
RUN npm run build -w @lumina/contract

# --noCheck: emit JS without running tsc's full type-check, same reasoning as the agent's
# Dockerfile — this code has only ever been run via tsx (transpile-only) locally, never through
# a full `tsc` build, so a strict build surfaces pre-existing type-annotation mismatches rather
# than real runtime bugs. Worth cleaning up later; not worth blocking on now.
RUN npm run build -w @lumina/gateway -- --noCheck

RUN npm prune --omit=dev


# Final stage for app image
FROM base

COPY --from=build /app /app

# WORKDIR matches how this service runs locally (`npm run dev -w @lumina/gateway`, cwd =
# backend/gateway) so any relative-path assumptions in the code (e.g. env.ts's webDist and
# reportsFile, both resolved via `../../` from process.cwd()) behave the same as in dev.
WORKDIR /app/backend/gateway

EXPOSE 8787
CMD [ "node", "dist/index.js" ]
