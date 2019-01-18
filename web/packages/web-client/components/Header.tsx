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
              onBlur={this.onProfileLeave}
              style={{ outline: 'none' }}
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
                  <span
                    style={{
                      boxShadow:
                        '0px 1px 3px 0px rgba(0,0,0,0.2), 0px 1px 1px 0px rgba(0,0,0,0.14), 0px 2px 1px -1px rgba(0,0,0,0.12)',
                      height: 60,
                      display: 'flex',
                      justifyContent: 'center',
                      alignItems: 'center',
                      position: 'absolute',
                      top: 10,
                      right: 10,
                      background: 'white'
                    }}
                  >
                    <a
                      style={{ padding: 14, cursor: 'pointer' }}
                      onClick={this.logout}
                    >
                      Logout
                    </a>
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
