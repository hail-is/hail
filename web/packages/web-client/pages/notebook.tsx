import { PureComponent } from 'react';
import auth from '../libs/auth';
import fetch from 'isomorphic-unfetch';
import getConfig from 'next/config';

// TODO: If user isn't logged in, and make it to this page
// after 401, tell them they need to log in
const cfg = getConfig().publicRuntimeConfig.NOTEBOOK;
const URL = cfg.URL;

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
  alive: notebook[];
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

const getAlive = (nbs: notebooks) =>
  Object.values(nbs).filter((d: notebook) => d.svc_status != kubeState.Deleted);

let resultsPromise: Promise<{ notebooks: notebooks; alive: notebook[] }>;

let notebooks: notebooks = {};
let alive: notebook[] = [];
let initialized = false;

const startRequest = () => {
  if (resultsPromise) {
    return resultsPromise;
  }

  resultsPromise = fetch(`${URL}/api`, {
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

      notebooks = data;
      alive = getAlive(data);

      return { notebooks, alive };
    })
    .finally(() => {
      initialized = true;
    });

  return resultsPromise;
};

let isListening: boolean;
const callbackFunctions: ((nb: notebooks, alive: notebook[]) => void)[] = [];

const startListener = () => {
  if (isListening) {
    return;
  }

  const name = URL.replace(/http/, 'ws'); //automatically wss if https
  const ws = new WebSocket(`${name}/api/ws`);

  isListening = true;

  ws.onmessage = ev => {
    const data: kubeWebsocketUpdate = JSON.parse(ev.data);

    const event = data.event;
    const updated = data.resource;

    const isDeleted =
      updated.svc_status === kubeState.Deleted ||
      updated.pod_status === kubeState.Deleted;

    if (!(event === kubeEvent.DELETED || isDeleted)) {
      notebooks[updated.pod_name] = updated;
      alive = getAlive(notebooks);

      // TODO: add errors
      callbackFunctions.forEach(cb => {
        cb(notebooks, alive);
      });

      return;
    }

    const existing = !!notebooks[updated.pod_name];

    if (!existing) {
      return;
    }

    callbackFunctions.forEach(cb => {
      cb(notebooks, alive);
    });

    delete notebooks[updated.pod_name];
    alive = getAlive(notebooks);

    // TODO: add errors
    callbackFunctions.forEach(cb => {
      cb(notebooks, alive);
    });
  };
};

const createNotebook = async () => {
  return fetch(`${URL}/api`, {
    headers: {
      Authorization: `Bearer ${auth.accessToken}`
    },
    method: 'POST'
  })
    .then(d => d.json())
    .then((notebook: notebook) => {
      notebooks[notebook.pod_name] = notebook;
      alive = getAlive(notebooks);

      return { notebooks, alive };
    });
};

const removeItemFromNotebook = (notebook: notebook) => {
  delete notebooks[notebook.pod_name];
  alive = getAlive(notebooks);
};

const deleteNotebook = (notebook: notebook, skipRemove: boolean) => {
  return fetch(`${URL}/api/${notebook.svc_name}`, {
    headers: {
      Authorization: `Bearer ${auth.accessToken}`
    },
    method: 'DELETE'
  }).then(response => {
    if (!response.ok && response.status !== 404) {
      console.error(response);
      throw Error(`${response.status}`);
    }

    if (!skipRemove) {
      delete notebooks[notebook.pod_name];
      alive = getAlive(notebooks);
    }

    return { notebooks, alive };
  });
};

let loadingTimeout: NodeJS.Timeout | null;
// TODO: decide whether we want to show only notebooks whose svc and pod status
// TODO: check that there are no side-effects for mutating this.state.notebooks
// are both Running
// Argument against this is to give fine-grained insight into what Kube is doing
// Because Kube is not a good queue, and will give out-of-order events
// which may be easier for the user to understand, that for us to present always as in-order
class Notebook extends PureComponent<any, state> {
  state: state = {
    loading: 0,
    unauthorized: false,
    notebooks: {},
    alive: []
  };

  constructor(props: props) {
    super(props);

    if (initialized === false) {
      this.state.loading = 1;

      if (typeof window !== 'undefined') {
        startRequest();
      }
    } else {
      this.state.notebooks = Object.assign({}, notebooks);
      this.state.alive = alive.slice(0);
    }
  }

  componentDidMount() {
    if (initialized) {
      loadingTimeout = setTimeout(() => {
        this.setState({
          loading: 1
        });
      }, 250);
    }

    startRequest().then(() => {
      if (loadingTimeout) {
        clearTimeout(loadingTimeout);
        loadingTimeout = null;
      }

      this.updateState(true);

      callbackFunctions.push(() => this.updateState(true));

      startListener();
    });
  }

  componentWillUnmount() {
    callbackFunctions.pop();
    if (loadingTimeout) {
      clearTimeout(loadingTimeout);
    }
  }

  createNotebook = () => {
    loadingTimeout = setTimeout(() => {
      this.setState({
        loading: 1
      });
    }, 33);

    createNotebook()
      .then(() => {
        if (loadingTimeout) {
          clearTimeout(loadingTimeout);
          loadingTimeout = null;
        }

        this.updateState(true);
      })
      .catch(() => {
        this.setState({ loading: -1 });
      });
  };

  // It is much, much faster to build a new array with the desired element
  // excluded, than to splice
  // https://jsperf.com/splice-vs-filter
  updateState = (success: boolean) => {
    this.setState({
      notebooks: Object.assign({}, notebooks),
      alive: alive.slice(0),
      loading: success ? 0 : -1
    });
  };

  deleteNotebook = (notebook: notebook) => {
    // Do the deletion opportunisticly
    removeItemFromNotebook(notebook);

    this.updateState(true);

    deleteNotebook(notebook, true).catch(() => {
      this.setState({ loading: -1 });
    });
  };

  render() {
    return (
      <div id="notebook" className="centered">
        {this.state.loading ? (
          this.state.loading === -1 ? (
            <span>Error</span>
          ) : (
            <div className="spinner" />
          )
        ) : !this.state.alive.length ? (
          <button onClick={this.createNotebook}>Create</button>
        ) : (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              cursor: 'pointer'
            }}
          >
            {this.state.alive.map((d: notebook) => (
              // Kubernetes is far too noisy, this prevents us from seeing pods that
              // are unreachable (since back-end/service is missing)
              <span
                key={d.pod_name}
                style={{
                  flexDirection: 'row',
                  display: 'flex',
                  alignItems: 'center',
                  marginBottom: '14px',
                  justifyContent: 'space-around',
                  minWidth: '320px'
                }}
              >
                {d.svc_status === kubeState.Running &&
                d.pod_status === kubeState.Running ? (
                  <i
                    style={{ color: '#428bca', marginRight: '14px' }}
                    className="material-icons"
                  >
                    done
                  </i>
                ) : (
                  <div className="spinner" style={{ marginRight: '14px' }} />
                )}
                <a
                  target="_blank"
                  style={{ flexDirection: 'column', display: 'flex' }}
                  href={`${URL}/instance/${d.svc_name}/?token=${d.token}`}
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
                  <span className="small">Created on: {d.creation_date}</span>
                </a>
                <i
                  className="material-icons link-button"
                  style={{ marginLeft: '56px' }}
                  onClick={() => this.deleteNotebook(d)}
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
