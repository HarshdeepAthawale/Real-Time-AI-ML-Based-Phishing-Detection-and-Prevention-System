FROM node:20-alpine AS builder
WORKDIR /app

# Copy shared source code (no separate npm install - will use service's node_modules)
COPY shared ./shared

# Copy detection-api package.json and install dependencies
COPY core-services/detection-api/package*.json ./core-services/detection-api/
WORKDIR /app/core-services/detection-api
RUN npm ci

# Copy shared package.json and merge dependencies (install into service node_modules)
WORKDIR /app
COPY shared/package.json ./shared/
RUN cd shared && npm install --no-save --prefix ../core-services/detection-api 2>/dev/null || true

# Copy detection-api source code
COPY core-services/detection-api ./core-services/detection-api

# Build detection-api (shared will be compiled as part of this)
WORKDIR /app/core-services/detection-api
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/core-services/detection-api/dist ./dist
COPY --from=builder /app/core-services/detection-api/node_modules ./node_modules
COPY --from=builder /app/core-services/detection-api/package.json ./
COPY --from=builder /app/shared ./shared
EXPOSE 3001
CMD ["node", "dist/index.js"]
