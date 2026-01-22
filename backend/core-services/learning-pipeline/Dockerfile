FROM node:20-alpine AS builder
WORKDIR /app

# Copy shared package.json and install dependencies
COPY shared/package*.json ./shared/
RUN cd shared && npm ci

# Copy learning-pipeline package.json and install dependencies
COPY core-services/learning-pipeline/package*.json ./core-services/learning-pipeline/
RUN cd core-services/learning-pipeline && npm ci

# Copy shared source code
COPY shared ./shared

# Copy learning-pipeline source code
COPY core-services/learning-pipeline ./core-services/learning-pipeline

# Build learning-pipeline (shared will be compiled as part of this)
WORKDIR /app/core-services/learning-pipeline
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/core-services/learning-pipeline/dist ./dist
COPY --from=builder /app/core-services/learning-pipeline/node_modules ./node_modules
COPY --from=builder /app/core-services/learning-pipeline/package.json ./
COPY --from=builder /app/shared ./shared
CMD ["node", "dist/index.js"]
