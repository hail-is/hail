const { createServer } = require('http');
const { parse } = require('url');
const next = require('next');
const fs = require('fs');

const isDev = process.env.NODE_ENV !== 'production';

const app = next({ dev: isDev });
const handle = app.getRequestHandler();

const favicon = fs.readFileSync('./static/favicon.ico');

const port = process.env.PORT || 3000;
app.prepare().then(() => {
  createServer((req, res) => {
    // true indicates parse the get query
    const parsedUrl = parse(req.url, true);

    if (parsedUrl.pathname === '/favicon.ico') {
      res.writeHeader(200, 'image/png');
      res.write(favicon);
      res.end();
    } else {
      handle(req, res, parsedUrl);
    }
  }).listen(port, err => {
    if (err) {
      throw err;
    }

    console.info(`Running on port ${port}`);
  });
});
