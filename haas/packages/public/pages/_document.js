// Server-rendered component, rendered only once, even in SPA mode
// A good place to inject site-wide <head> tags </head>
import Document, { Head, Main, NextScript } from 'next/document';

class MyDocument extends Document {
  render() {
    // const { pageContext } = this.props;
    // const isDark = this.props.pageContext.darkTheme;

    return (
      <html lang="en" dir="ltr">
        <Head>
          <meta charSet="utf-8" />
          {/* Use minimum-scale=1 to enable GPU rasterization */}
          <meta
            name="viewport"
            content="minimum-scale=1, initial-scale=1, width=device-width, shrink-to-fit=no"
          />
          {/* <link
            rel="shortcut icon"
            type="image/x-icon"
            href="/static/favicon.ico"
          /> */}
          <link
            rel="stylesheet"
            href="https://fonts.googleapis.com/icon?family=Material+Icons"
          />
        </Head>
        <body style={{ margin: 0 }}>
          <Main />
          <NextScript />
        </body>
      </html>
    );
  }
}

export default MyDocument;
