# syntax=docker/dockerfile:1
# Build from apps/web:  docker build -t interview-web apps/web
# Behind a TLS-intercepting proxy, pass its CA bundle:  --secret id=extra_ca,src=/path/ca.pem
FROM node:22-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
COPY scripts scripts
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then export NODE_EXTRA_CA_CERTS=/run/secrets/extra_ca; fi; \
    npm ci
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
