FROM mhart/alpine-node:10

WORKDIR /web
COPY . .

RUN npm install
RUN npm run build

EXPOSE 3000