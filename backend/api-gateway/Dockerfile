FROM node:20-alpine AS builder
WORKDIR /app

# Copy shared package.json and install dependencies
COPY shared/package*.json ./shared/
RUN cd shared && npm ci

# Copy api-gateway package.json and install dependencies  
COPY api-gateway/package*.json ./api-gateway/
RUN cd api-gateway && npm ci

# Copy shared source code
COPY shared ./shared

# Copy api-gateway source code
COPY api-gateway ./api-gateway

# Build api-gateway (shared will be compiled as part of this)
WORKDIR /app
RUN cd api-gateway && npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/api-gateway/dist ./dist
COPY --from=builder /app/api-gateway/node_modules ./node_modules
COPY --from=builder /app/api-gateway/package.json ./
COPY --from=builder /app/shared ./shared
EXPOSE 3000
CMD ["node", "dist/index.js"]
