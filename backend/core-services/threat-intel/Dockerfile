FROM node:20-alpine AS builder
WORKDIR /app

# Copy shared package.json and install dependencies
COPY shared/package*.json ./shared/
RUN cd shared && npm ci

# Copy threat-intel package.json and install dependencies
COPY core-services/threat-intel/package*.json ./core-services/threat-intel/
RUN cd core-services/threat-intel && npm ci

# Copy shared source code
COPY shared ./shared

# Copy threat-intel source code
COPY core-services/threat-intel ./core-services/threat-intel

# Build threat-intel (shared will be compiled as part of this)
WORKDIR /app/core-services/threat-intel
RUN npm run build

FROM node:20-alpine
WORKDIR /app
# Install curl for health checks
RUN apk add --no-cache curl
COPY --from=builder /app/core-services/threat-intel/dist ./dist
COPY --from=builder /app/core-services/threat-intel/node_modules ./node_modules
COPY --from=builder /app/core-services/threat-intel/package.json ./
COPY --from=builder /app/shared ./shared
EXPOSE 3002
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:3002/health || exit 1
CMD ["node", "dist/index.js"]
