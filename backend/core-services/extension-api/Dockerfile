FROM node:20-alpine AS builder
WORKDIR /app

# Copy shared source code
COPY shared ./shared

# Copy extension-api package.json and install dependencies
COPY core-services/extension-api/package*.json ./core-services/extension-api/
WORKDIR /app/core-services/extension-api
RUN npm ci

# Copy extension-api source code (into current workdir)
COPY core-services/extension-api ./

# Symlink so shared resolves deps from extension-api's node_modules
RUN ln -sf ../core-services/extension-api/node_modules /app/shared/node_modules

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
