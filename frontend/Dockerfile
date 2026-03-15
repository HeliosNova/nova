FROM node:20-slim AS base
WORKDIR /app

# Install dependencies first (cache layer)
COPY package.json package-lock.json ./
RUN npm ci

# Copy source
COPY . .

# ── Dev target (default) ──
FROM base AS dev
EXPOSE 5173
CMD ["npx", "vite", "--host", "0.0.0.0"]

# ── Production build ──
FROM base AS build
RUN npm run build

FROM nginx:alpine AS prod
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
