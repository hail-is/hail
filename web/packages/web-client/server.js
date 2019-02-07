const { createServer } = require('http');
const { parse } = require('url');
const next = require('next');
const fs = require('fs');

const isDev = process.env.NODE_ENV !== 'production';

const app = next({ dev: isDev });
const handle = app.getRequestHandler();

const favicon = fs.readFileSync('./static/favicon.ico');

// const cache = {};

// const send = (res, html) => {
//   res.writeHeader(200, 'text/html');
//   res.write(html);
//   res.end();
// };

// const pagesToCache = {
//   //Breaks dark mode: '/': isDev ? 0 : -1, // forever
//   // '/scorecard': isDev ? 0 : 3 * 60 * 1000, // 3 min
//   // '/scorecard/user': isDev ? 0 : 3 * 60 * 1000
// };

// const render = (cacheTime = 5 * 60 * 1000, req, res, pagePath, query) => {
//   if (
//     !!cache[req.url] &&
//     (cache[req.url].expires === -1 || cache[req.url].expires >= Date.now())
//   ) {
//     send(res, cache[req.url].html);
//     return;
//   }

//   app
//     .renderToHTML(req, res, pagePath, query)
//     .then(html => {
//       cache[req.url] = {
//         html,
//         expires: cacheTime === -1 ? cacheTime : cacheTime + Date.now()
//       };

//       send(res, html);
//     })
//     .catch(err => console.error(err));
// };

app.prepare().then(() => {
  createServer((req, res) => {
    // true indicates parse the get query
    const parsedUrl = parse(req.url, true);

    if (parsedUrl.pathname === '/favicon.ico') {
      res.writeHeader(200, 'image/png');
      res.write(favicon);
      res.end();
    }
    // else if (pagesToCache[parsedUrl.pathname]) {
    //   render(
    //     pagesToCache[parsedUrl.pathname],
    //     req,
    //     res,
    //     parsedUrl.pathname,
    //     parsedUrl.query
    //   );
    // }
    else {
      handle(req, res, parsedUrl);
    }
  }).listen(process.env.PORT || 3000, err => {
    if (err) {
      throw err;
    }

    console.info('Running on port 3000');
  });
});
