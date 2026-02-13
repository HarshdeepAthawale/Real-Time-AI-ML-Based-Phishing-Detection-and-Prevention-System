FROM node:20-alpine AS builder
WORKDIR /app

# Copy shared source code
COPY shared ./shared

# Copy sandbox-service package.json and install dependencies
COPY core-services/sandbox-service/package*.json ./core-services/sandbox-service/
WORKDIR /app/core-services/sandbox-service
RUN npm ci

# Copy sandbox-service source code (into current workdir)
COPY core-services/sandbox-service ./

# Symlink so shared resolves deps from sandbox-service's node_modules
RUN ln -sf ../core-services/sandbox-service/node_modules /app/shared/node_modules

# Build sandbox-service (shared will be compiled as part of this)
WORKDIR /app/core-services/sandbox-service
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/core-services/sandbox-service/dist ./dist
COPY --from=builder /app/core-services/sandbox-service/node_modules ./node_modules
COPY --from=builder /app/core-services/sandbox-service/package.json ./
COPY --from=builder /app/shared ./shared
EXPOSE 3004
CMD ["node", "dist/index.js"]
