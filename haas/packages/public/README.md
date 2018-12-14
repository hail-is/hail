[NextJS documentation](https://github.com/zeit/next.js)

TL;DR

```sh
# Compile
npm run build
# Run
npm run start
# If dev (slow, debug)
npm run dev
```

## Development

Install NodeJS >-= 10.14.1 LTS https://nodejs.org/en/

```sh
npm install -g yarn
cd haas/packages/public
yarn install

# Run web app in dev mode on port 3000
npm run dev
```

# TODO

1. Remove superagent from auth0 library. adds ~20KB to bundle
