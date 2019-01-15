FROM mhart/alpine-node:10

WORKDIR /app
COPY . .

RUN npm install && npm run bootstrap && npm run start