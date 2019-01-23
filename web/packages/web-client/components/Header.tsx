import { PureComponent, Fragment } from 'react';
import Link from 'next/link';
import {
  logout,
  addListener,
  removeListener,
  isAuthenticated
} from '../libs/auth';
import './Header/header.scss';
import { withRouter } from 'next/router';

const bStyle = 'link-button';

let listenerID: number;
class Header extends PureComponent {
  state: any = {
    showProfileControls: false,
    isLoggedIn: false
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

  onLogout() {
    logout();
  }

  constructor(props: any) {
    super(props);

    this.state.isLoggedIn = isAuthenticated();
  }

  componentDidMount() {
    listenerID = addListener(state => {
      if (this.state.isLoggedIn !== !!state.user) {
        console.info('changed');
        this.setState({ isLoggedIn: !!state.user });
      }
    });
  }

  componentWillUnmount() {
    removeListener(listenerID);
  }

  render() {
    // TODO: Figure out why typing information not being read
    const {
      router: { pathname }
    } = this.props as any;
    // interestingly, for initial page load, using this.state.isLoggedIn within the template
    // actually may result in a flash of incorrect state
    // const loggedIn = this.state.isLoggedIn;

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
        <span id="profile-divider" />
        {this.state.isLoggedIn ? (
          <Fragment>
            <Link href="/user" prefetch>
              <a className={`${bStyle}`} href="/">
                Profile
              </a>
            </Link>
            <button className={`${bStyle}`} onClick={this.onLogout}>
              Log out
            </button>
            {/* TODO: Add back in for narrow views
              // <span id="narrow-view" style={{ marginLeft: 'auto' }}>
              //   <span
              //     tabIndex={0}
              //     style={{ outline: 'none' }}
              //     onBlur={this.onProfileLeave}
              //   >
              //     <a className="icon-button" onClick={this.onProfileHover}>
              //       <i className="material-icons">face</i>
              //     </a>
              //     <span>
              //       {this.state.showProfileControls && (
              //         <span id="profile-menu">
              //           <a onClick={this.logout}>Logout</a>
              //         </span>
              //       )}
              //     </span>
              //   </span>
              // </span>
            */}
          </Fragment>
        ) : (
          <Link href="/login" prefetch>
            <a className={`${bStyle} ${pathname === '/login' ? 'active' : ''}`}>
              Login
            </a>
          </Link>
        )}
      </span>
    );
  }
}

export default withRouter(Header);
