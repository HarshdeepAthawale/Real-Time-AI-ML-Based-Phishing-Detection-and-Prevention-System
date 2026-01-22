FROM node:20-alpine AS builder
WORKDIR /app

# Copy shared package.json and install dependencies
COPY shared/package*.json ./shared/
RUN cd shared && npm ci

# Copy extension-api package.json and install dependencies
COPY core-services/extension-api/package*.json ./core-services/extension-api/
RUN cd core-services/extension-api && npm ci

# Copy shared source code
COPY shared ./shared

# Copy extension-api source code
COPY core-services/extension-api ./core-services/extension-api

# Build extension-api (shared will be compiled as part of this)
WORKDIR /app/core-services/extension-api
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/core-services/extension-api/dist ./dist
COPY --from=builder /app/core-services/extension-api/node_modules ./node_modules
COPY --from=builder /app/core-services/extension-api/package.json ./
COPY --from=builder /app/shared ./shared
EXPOSE 3003
CMD ["node", "dist/index.js"]
