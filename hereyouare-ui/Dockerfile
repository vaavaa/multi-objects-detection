# Сборка Vite/React (относительные URL к API для прокси через nginx)
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci || npm install
COPY . .
ENV VITE_HEREYOUARE_URL=
RUN npm run build

# Раздача статики + прокси к hereyouare backend
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
