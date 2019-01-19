const autoprefixer = require('autoprefixer');

module.exports = {
  plugins: [
    autoprefixer({
      browsers: [
        // your supported browser config here.
        'last 2 versions'
        // 'no ie <= 11'
      ],
      flexbox: true
    })
  ]
};
