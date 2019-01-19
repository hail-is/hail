import { PureComponent } from 'react';
import Link from 'next/link';
import auth from '../libs/auth';
import './Header/header.scss';
import Router, { withRouter } from 'next/router';

const bStyle = 'link-button';

export interface HeaderProps {
  authState: any;
}

class Header extends PureComponent<HeaderProps> {
  state: any = {
    showProfileControls: false
  };

  onProfileHover = () => {
    this.setState({
      showProfileControls: true
    });
  };

  onProfileLeave = () => {
    this.setState({
      showProfileControls: false
    });
  };

  logout = () => {
    auth.logout();
    this.setState({
      showProfileControls: false
    });
    Router.replace('/');
  };

  render() {
    // TODO: Figure out why typing information not being read
    const {
      authState,
      router: { pathname }
    } = this.props as any;

    return (
      <span id="header">
        <Link href="/">
          <a className={`${bStyle} ${pathname === '/' ? 'active' : ''}`}>
            <b>/</b>
          </a>
        </Link>
        <Link href="/notebook" prefetch>
          <a
            className={`${bStyle} ${pathname === '/notebook' ? 'active' : ''}`}
          >
            Notebook
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
          {authState.user ? (
            <span
              tabIndex={0}
              style={{ outline: 'none' }}
              onBlur={this.onProfileLeave}
            >
              <a
                className="icon-button"
                style={{ padding: 14 }}
                onClick={this.onProfileHover}
              >
                <i className="material-icons">face</i>
              </a>
              <span>
                {this.state.showProfileControls && (
                  <span id="profile-menu">
                    <a onClick={this.logout}>Logout</a>
                  </span>
                )}
              </span>
            </span>
          ) : (
            <Link href="/login" prefetch>
              <a
                className={`${bStyle} ${pathname === '/login' ? 'active' : ''}`}
              >
                Login
              </a>
            </Link>
          )}
        </span>
      </span>
    );
  }
}

export default withRouter(Header);
