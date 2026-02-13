FROM node:20-alpine AS builder
WORKDIR /app

# Copy shared source code (no separate npm install - will use service's node_modules)
COPY shared ./shared

# Copy detection-api package.json and install dependencies
COPY core-services/detection-api/package*.json ./core-services/detection-api/
WORKDIR /app/core-services/detection-api
RUN npm ci

# Copy detection-api source code (into current workdir)
COPY core-services/detection-api ./

# Symlink so shared resolves deps from detection-api's node_modules (single typeorm)
RUN ln -sf ../core-services/detection-api/node_modules /app/shared/node_modules

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
