import { PureComponent } from 'react';
import Link from 'next/link';
// import Auth from 'libs/Auth';
import './Header/header.scss';
import { withRouter } from 'next/router';

const bStyle = 'link-button';
class Header extends PureComponent {
  render() {
    // TODO: Figure out why typing information not being read
    const {
      router: { pathname }
    } = this.props as any;

    return (
      <span id="header">
        <Link href="/">
          <a className={`${bStyle} ${pathname === '/' ? 'active' : ''}`}>
            <b>/</b>
          </a>
        </Link>
        <Link href="/scorecard" prefetch>
          <a
            className={`${bStyle} ${pathname === '/scorecard' ? 'active' : ''}`}
          >
            Scorecard
          </a>
        </Link>
        <span style={{ marginLeft: 'auto' }}>
          <Link href="/login" prefetch>
            <a className={`${bStyle} ${pathname === '/login' ? 'active' : ''}`}>
              Login
            </a>
          </Link>
        </span>
      </span>
    );
  }
}

export default withRouter(Header);
