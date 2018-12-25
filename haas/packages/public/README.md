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

### Testing SSL

[Instruction on how to trust a self signed certificate](https://www.accuweaver.com/2014/09/19/make-chrome-accept-a-self-signed-certificate-on-osx/)

```sh
sudo ./self-signed-cert.sh

cp local-nginx-conf.conf /path/to/nginx/servers/

npm run build
npm run prod-test-https
```

# TODO

1. Remove superagent from auth0 library. adds ~20KB to bundle
