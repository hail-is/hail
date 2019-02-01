import { PureComponent } from 'react';
import auth from '../libs/auth';
import fetch from 'isomorphic-unfetch';
import getConfig from 'next/config';

const DOMAIN = getConfig().publicRuntimeConfig.NOTEBOOK.DOMAIN;

declare type notebook = {
  svc_name: string;
  name: string;
  pod_name: string;
  token: string;
  creation_date: string;
  svc_status: string;
  pod_status: string; // The kubernetes status.phase val for pod
};

declare type notebooks = { [pod_name: string]: notebook };

declare type props = {
  pageProps: {
    notebooks: notebooks;
  };
};

declare type state = {
  unauthorized: boolean;
  notebooks: notebooks;
  loading: number; //-1, 0, 1: failed, not, loading
};

enum kubeState {
  Deleted = 'Deleted',
  Running = 'Running'
}

enum kubeEvent {
  ADDED = 'ADDDED',
  DELETED = 'DELETED',
  MODIFIED = 'MODIFIED'
}

declare type kubeWebsocketUpdate = {
  event: kubeEvent;
  resource: notebook;
};

// TODO: decide whether we want to show only notebooks whose svc and pod status
// TODO: check that there are no side-effects for mutating this.state.notebooks
// are both Running
// Argument against this is to give fine-grained insight into what Kube is doing
// Because Kube is not a good queue, and will give out-of-order events
// which may be easier for the user to understand, that for us to present always as in-order
class Notebook extends PureComponent<props, state> {
  state: state = {
    loading: 0,
    unauthorized: false,
    notebooks: {}
  };

  static async getInitialProps() {
    let notebooks: notebooks = {};
    try {
      notebooks = await fetch(`${DOMAIN}/api`, {
        headers: {
          Authorization: `Bearer ${auth.accessToken}`
        }
      })
        .then(d => d.json())
        .then((aNotebooks: notebook[]) => {
          const data: notebooks = {};

          aNotebooks.forEach(n => {
            data[n.pod_name] = n;
          });

          return data;
        });
    } finally {
      return { pageProps: { notebooks } };
    }
  }

  constructor(props: props) {
    super(props);

    this.state.notebooks = props.pageProps.notebooks;
  }

  componentDidMount() {
    const name = DOMAIN.replace(/http/, 'ws'); //automatically wss if https

    const ws = new WebSocket(`${name}/api/ws?access_token=${auth.accessToken}`);

    ws.onmessage = ev => {
      const data: kubeWebsocketUpdate = JSON.parse(ev.data);

      const event = data.event;
      const updated = data.resource;

      const isDeleted =
        updated.svc_status === kubeState.Deleted ||
        updated.pod_status === kubeState.Deleted;

      if (!(event === kubeEvent.DELETED || isDeleted)) {
        this.setState((p: state) => {
          const notebooks = Object.assign({}, p.notebooks);
          notebooks[updated.pod_name] = updated;
          return { notebooks };
        });

        return;
      }

      const existing = !!this.state.notebooks[updated.pod_name];

      if (!existing) {
        return;
      }

      this.setState((p: state) => {
        const notebooks = Object.assign({}, p.notebooks);
        delete notebooks[updated.pod_name];
        return { notebooks };
      });
    };
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

      this.state.notebooks[notebook.pod_name] = notebook;
      // prevState should not be mutated
      // fastest shallow copy method is .slice()
      // https://jsperf.com/new-array-vs-splice-vs-slice/31
      this.setState((p: state) => {
        const notebooks = Object.assign({}, p.notebooks);

        notebooks[notebook.pod_name] = notebook;
        return {
          notebooks,
          loading: 0
        };
      });
    } catch (e) {
      this.setState({ loading: -1 });
    }
  };

  deleteNotebook = async (notebook: notebook) => {
    this.setState({ loading: 1 });

    try {
      const response = await fetch(`${DOMAIN}/api/${notebook.svc_name}`, {
        headers: {
          Authorization: `Bearer ${auth.accessToken}`
        },
        method: 'DELETE'
      });

      // TODO: This isn't catching 404 or other errors
      if (!response.ok && response.status !== 404) {
        this.setState({ loading: -1 });
      }

      this.removeNotebook(notebook);
    } catch (e) {
      this.setState({ loading: -1 });
    }

    // It is much, much faster to build a new array with the desired element
    // excluded, than to splice
    // https://jsperf.com/splice-vs-filter
  };

  removeNotebook = (n: notebook) => {
    this.setState((p: state) => {
      const notebooks = Object.assign({}, this.state.notebooks);

      delete notebooks[n.pod_name];
      return { notebooks, loading: 0 };
    });
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
        ) : !Object.keys(this.state.notebooks).length ? (
          <button onClick={this.createNotebook}>Create</button>
        ) : (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              cursor: 'pointer'
            }}
          >
            {Object.values(this.state.notebooks).map(
              (d: notebook) =>
                // Kubernetes is far too noisy, this prevents us from seeing pods that
                // are unreachable (since back-end/service is missing)
                d.svc_status !== kubeState.Deleted && (
                  <span
                    key={d.pod_name}
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
                      href={`${DOMAIN}/instance/${d.svc_name}/?authorization=${
                        auth.accessToken
                      }&token=${d.token}`}
                    >
                      <b>{d.name}</b>
                      <span className="small">
                        Service Name: <b>{d.svc_name}</b>
                      </span>
                      <span className="small">
                        Pod Name: <b>{d.pod_name}</b>
                      </span>
                      <span className="small">
                        Service: <b>{d.svc_status}</b>
                      </span>
                      <span className="small">
                        Pod: <b>{d.pod_status}</b>
                      </span>
                      <span className="small">
                        Created on: {d.creation_date}
                      </span>
                    </a>
                    <i
                      className="material-icons link-button"
                      style={{ marginLeft: '14px' }}
                      onClick={() => this.deleteNotebook(d)}
                    >
                      close
                    </i>
                  </span>
                )
            )}
          </div>
        )}
      </div>
    );
  }
}

export default Notebook;
