# Build from apps/web:  docker build -t interview-web apps/web
FROM node:22-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
COPY scripts scripts
RUN npm ci
COPY . .
RUN npm run build

FROM node:22-slim
ENV NODE_ENV=production PORT=3000 HOSTNAME=0.0.0.0
WORKDIR /app
RUN useradd --system --uid 10001 web
COPY --from=build --chown=web /app/.next/standalone ./
COPY --from=build --chown=web /app/.next/static ./.next/static
COPY --from=build --chown=web /app/public ./public
USER web
EXPOSE 3000
CMD ["node", "server.js"]
