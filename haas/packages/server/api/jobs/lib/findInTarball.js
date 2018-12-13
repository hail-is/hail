const tar = require('tar-stream');
const fs = require('fs');
const zlib = require('zlib');

module.exports = (tarballPath, innerFileName, cb) => {
  const extract = tar.extract();
  let data = '';
  cb()
  console.info('extracting', tarballPath, innerFileName);
  extract.on('entry', (header, stream, innerCb) => {
    stream.on('data', (chunk) => {
      // console.info('header name', header.name);
      if (header.name == innerFileName) {
        console.info("equals");
        data += chunk;
      }
    });

    stream.on('end', () => innerCb());

    stream.resume();
  });

  extract.on('finish', () => cb(data));

  fs.createReadStream(tarballPath).pipe(zlib.createGunzip()).pipe(extract);
};