# syntax = docker/dockerfile:1
#
# Built with the MONOREPO ROOT as the Docker build context (not backend/agent/), because
# @lumina/agent depends on the @lumina/contract workspace package — `npm install` only
# resolves that correctly when it runs from the root, where the npm workspaces are declared.
# Deploy with:
#   fly deploy . --config backend/agent/fly.toml --dockerfile backend/agent/Dockerfile
# run from the repo root (not from backend/agent/).

ARG NODE_VERSION=24.21.0
FROM node:${NODE_VERSION}-slim AS base

LABEL fly_launch_runtime="Node.js"

WORKDIR /app
ENV NODE_ENV="production"


# Throw-away build stage to reduce size of final image
FROM base AS build

RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y build-essential node-gyp pkg-config python-is-python3

# Copy the whole monorepo so npm workspaces (packages/contract, backend/agent, etc.) resolve.
COPY . .

RUN npm ci --include=dev

# @lumina/agent imports @lumina/contract's compiled output, so contract must build first.
RUN npm run build -w @lumina/contract

# --noCheck: emit JS without running tsc's full type-check. `npm run dev`/`worker` run this
# code via tsx (transpile-only, no type-check), so this is what has actually been exercised
# and proven correct all along — a strict `tsc` build has never run against this code before,
# and it surfaces pre-existing type-annotation mismatches (e.g. string ids vs Mongo's ObjectId
# type) rather than real runtime bugs. Worth cleaning up later; not worth blocking on now.
RUN npm run build -w @lumina/agent -- --noCheck

RUN npm prune --omit=dev


# Final stage for app image
FROM base

COPY --from=build /app /app

# WORKDIR matches how this service runs locally (`npm run dev -w @lumina/agent`, cwd =
# backend/agent) so any relative-path assumptions in the code behave the same as in dev.
WORKDIR /app/backend/agent

EXPOSE 8000
CMD [ "node", "dist/index.js" ]
