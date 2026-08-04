# Frontend image: Vite build -> static files served by nginx.
#
# VITE_API_BASE_URL defaults to "/api/v1" here, not an absolute URL. Vite
# inlines env vars at BUILD time, so an absolute "http://localhost:8000/..."
# would be burned into the bundle and break the moment the app is served from
# anywhere other than the developer's own machine. A relative path means the
# browser calls the same origin it loaded the page from, nginx proxies it to
# the backend, and CORS never enters the picture.
#
# Build from the PROJECT ROOT:
#   docker build -f frontend/Dockerfile -t sdp-frontend .

# Debian-slim rather than Alpine for the BUILD stage. This stage is discarded
# -- only `dist/` survives into the nginx image -- so its base costs nothing
# in the shipped artefact, and glibc avoids the musl-specific optional-native-
# dependency problems Rollup and esbuild are prone to on Alpine.
FROM node:24-slim AS build

WORKDIR /app

# package files first so `npm ci` is cached independently of source edits.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

ARG VITE_API_BASE_URL=/api/v1
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build


FROM nginx:1.27-alpine AS runtime

# Rendered by the base image's 20-envsubst-on-templates.sh at container
# start, which is how ${SDP_API_KEY} gets injected without baking it into the
# image. NGINX_ENVSUBST_FILTER restricts substitution to SDP_-prefixed vars so
# envsubst cannot eat nginx's own $host / $scheme / $request_uri.
ENV NGINX_ENVSUBST_FILTER=^SDP_
COPY frontend/default.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1
