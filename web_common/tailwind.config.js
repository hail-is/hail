/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./web_common/**/*.{html,js}",
    "../auth/**/*.{html,js}",
    "../batch/**/*.{html,js}",
  ],
  theme: {
    extend: {
      animation: {
        'spin': 'spin 1.5s linear infinite',
      }
    },
  },
  plugins: [],
}
