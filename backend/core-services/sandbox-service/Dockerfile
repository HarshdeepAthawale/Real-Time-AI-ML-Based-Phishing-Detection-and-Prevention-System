FROM node:20-alpine AS builder
WORKDIR /app

# Copy shared package.json and install dependencies
COPY shared/package*.json ./shared/
RUN cd shared && npm ci

# Copy sandbox-service package.json and install dependencies
COPY core-services/sandbox-service/package*.json ./core-services/sandbox-service/
RUN cd core-services/sandbox-service && npm ci

# Copy shared source code
COPY shared ./shared

# Copy sandbox-service source code
COPY core-services/sandbox-service ./core-services/sandbox-service

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
