FROM node:20-alpine AS builder
WORKDIR /app

# Copy shared source code
COPY shared ./shared

# Copy learning-pipeline package.json and install dependencies
COPY core-services/learning-pipeline/package*.json ./core-services/learning-pipeline/
WORKDIR /app/core-services/learning-pipeline
RUN npm ci

# Copy learning-pipeline source code (into current workdir)
COPY core-services/learning-pipeline ./

# Symlink so shared resolves deps from learning-pipeline's node_modules
RUN ln -sf ../core-services/learning-pipeline/node_modules /app/shared/node_modules

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
