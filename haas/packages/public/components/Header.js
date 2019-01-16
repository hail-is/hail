import { PureComponent, Fragment } from 'react';
import PropTypes from 'prop-types';

import Link from 'next/link';
import Router, { withRouter } from 'next/router';
import LoginLink from 'components/Login/Link';

import classNames from 'classnames';

import 'components/Header/header.scss';

class Header extends PureComponent {
  state = {
    menuOpen: false
  };

  handleLogout = () => {
    this.props.auth.logout();
    this.setState({ menuOpen: false });
    Router.replace('/');
  };

  // Render does not get (props, state) passed to it
  render() {
    const { pathname } = this.props.router;

    const isDark = 'false';
    return (
      <span id="Header">
        <div id="appBar">
          <Link href="/" passHref>
            <a
              href="/"
              label="Home"
              id="home"
              className={pathname === '/' ? 'is-active' : null}
            >
              H
            </a>
          </Link>
          <Link href="/notebook" passHref prefetch>
            <a
              aria-label="Notebook Service"
              className={classNames('link-button', {
                'is-active': pathname === '/notebook'
              })}
            >
              Notebook
            </a>
          </Link>

          <Link href="/scorecard" passHref prefetch>
            <a
              aria-label="Scorecard"
              className={classNames('link-button', {
                'is-active': pathname === '/scorecard'
              })}
            >
              Scorecard
            </a>
          </Link>

          <Link href="/tutorial" passHref>
            <a
              aria-label="Tutorial"
              className={classNames('link-button', {
                'is-active': pathname === '/tutorial'
              })}
            >
              Tutorial
            </a>
          </Link>

          <span
            style={{ marginLeft: 'auto', outline: 'none' }}
            onBlur={() => this.setState({ menuOpen: false })}
            tabIndex="0"
          >
            {this.props.auth.state.user ? (
              <span
                aria-label="User panel"
                aria-haspopup="true"
                onClick={() => this.setState({ menuOpen: true })}
                color="inherit"
                style={{ marginTop: -5 }}
                className="link-button"
              >
                <i className="material-icons">account_circle</i>
                {this.state.menuOpen && (
                  <div id="header-user-menu" onClick={this.handleLogout}>
                    Log out
                  </div>
                )}
              </span>
            ) : (
              <Link href="/login" passHref>
                <a
                  aria-label="Login"
                  className={classNames('link-button', {
                    'is-active': pathname === '/login'
                  })}
                >
                  Login
                </a>
              </Link>
            )}
          </span>
        </div>
      </span>
    );
  }
}

export { LoginLink };
export default withRouter(Header);
