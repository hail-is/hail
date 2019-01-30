import { PureComponent } from 'react';
import auth from '../libs/auth';
import fetch from 'isomorphic-unfetch';
import getConfig from 'next/config';

const DOMAIN = getConfig().publicRuntimeConfig.NOTEBOOK.DOMAIN;

declare type notebook = string[3];

declare type props = {
  pageProps: {
    notebooks: notebook[];
  };
};

declare type state = {
  unauthorized: boolean;
  notebooks: notebook[];
  loading: number; //-1, 0, 1: failed, not, loading
};

class Notebook extends PureComponent<props, state> {
  state: state = {
    loading: 0,
    unauthorized: false,
    notebooks: []
  };

  // TODO: log errors
  static async getInitialProps() {
    let notebooks: notebook[] = [];
    try {
      notebooks = await fetch(`${DOMAIN}/api`, {
        headers: {
          Authorization: `Bearer ${auth.accessToken}`
        }
      }).then(d => d.json());
    } finally {
      return { pageProps: { notebooks } };
    }
  }

  constructor(props: props) {
    super(props);
    this.state.notebooks = props.pageProps.notebooks;
  }

  createNotebook = async () => {
    this.setState({ loading: 1 });

    const formData = new FormData();
    formData.append('image', 'hail');

    try {
      const notebook: notebook = await fetch(`${DOMAIN}/api`, {
        headers: {
          Authorization: `Bearer ${auth.accessToken}`
        },
        method: 'POST',
        body: formData
      }).then(d => d.json());

      // prevState should not be mutated
      // fastest shallow copy method is .slice()
      // https://jsperf.com/new-array-vs-splice-vs-slice/31
      this.setState((prevState: state) => {
        const notebooks = prevState.notebooks.slice();
        notebooks.push(notebook);

        return {
          notebooks,
          loading: 0
        };
      });
    } catch (e) {
      console.error(e);
      this.setState({ loading: -1 });
    }
  };

  deleteNotebook = async (notebook: notebook, excisedIndex: number) => {
    this.setState({ loading: 1 });

    try {
      await fetch(`${DOMAIN}/api/${notebook[2]}`, {
        headers: {
          Authorization: `Bearer ${auth.accessToken}`
        },
        method: 'DELETE'
      });

      this.setState((prevState: state) => ({
        loading: 0,
        notebooks: prevState.notebooks.filter(
          (_, idx: number) => idx !== excisedIndex
        )
      }));
    } catch (e) {
      console.error(e);
      this.setState({ loading: -1 });
    }

    // It is much, much faster to build a new array with the desired element
    // excluded, than to splice
    // https://jsperf.com/splice-vs-filter
  };

  render() {
    return (
      <div id="notebook" className="centered">
        {this.state.loading ? (
          this.state.loading === -1 ? (
            <span>Error</span>
          ) : (
            <span>Loading...</span>
          )
        ) : !this.state.notebooks.length ? (
          <button onClick={this.createNotebook}>Create</button>
        ) : (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              cursor: 'pointer'
            }}
          >
            {this.state.notebooks.map((d: any, idx: number) => (
              <span
                key={idx}
                style={{
                  flexDirection: 'row',
                  display: 'flex',
                  alignItems: 'center',
                  marginBottom: '14px',
                  justifyContent: 'space-between',
                  minWidth: '33vw'
                }}
              >
                <a
                  style={{ flexDirection: 'column', display: 'flex' }}
                  href={`${DOMAIN}/instance/${d[1]}/?authorization=${
                    auth.accessToken
                  }&token=${d[2]}`}
                >
                  <b>{d[0]}</b>
                  <span className="small">{d[1]}</span>
                </a>
                <i
                  className="material-icons link-button"
                  style={{ marginLeft: '14px' }}
                  onClick={() => this.deleteNotebook(d, idx)}
                >
                  close
                </i>
              </span>
            ))}
          </div>
        )}
      </div>
    );
  }
}

export default Notebook;
