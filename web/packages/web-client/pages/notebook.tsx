import { PureComponent } from 'react';

import {
  Notebook,
  URL,
  startRequest,
  callbackFunctions,
  startListener,
  createNotebook,
  removeItemFromNotebook,
  deleteNotebook,
  notebookType,
  notebooksType,
  kubeState
} from '../components/Notebook/datastore';

import '../styles/pages/notebook.scss';

declare type props = {
  pageProps: {
    notebooks: notebookType;
  };
};

declare type state = {
  unauthorized: boolean;
  notebooks: notebooksType;
  alive: notebookType[];
  loading: number; //-1, 0, 1: failed, not, loading
};

const isServer = typeof window === 'undefined';

let loadingTimeout: NodeJS.Timeout | null;

// TODO: decide whether we want to show only notebooks whose svc and pod status
// TODO: check that there are no side-effects for mutating this.state.notebooks
// are both Running
// Argument against this is to give fine-grained insight into what Kube is doing
// Because Kube is not a good queue, and will give out-of-order events
// which may be easier for the user to understand, that for us to present always as in-order
class NotebookPage extends PureComponent<any, state> {
  state: state = {
    loading: 0,
    unauthorized: false,
    notebooks: {},
    alive: []
  };

  constructor(props: props) {
    super(props);

    if (Notebook.initialized === false) {
      this.state.loading = 1;

      if (!isServer) {
        startRequest();
      }
    } else {
      this.state.notebooks = Object.assign({}, Notebook.notebooks);
      this.state.alive = Notebook.aliveNotebooks.slice(0);
    }
  }

  componentDidMount() {
    if (Notebook.initialized) {
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
      notebooks: Object.assign({}, Notebook.notebooks),
      alive: Notebook.aliveNotebooks.slice(0),
      loading: success ? 0 : -1
    });
  };

  deleteNotebook = (notebook: notebookType) => {
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
            {this.state.alive.map((d: notebookType) => (
              // Kubernetes is far too noisy, this prevents us from seeing pods that
              // are unreachable (since back-end/service is missing)
              <span key={d.pod_name} className={'nb'}>
                {d.svc_status === kubeState.Running &&
                d.pod_status === kubeState.Running &&
                d.condition &&
                d.condition.status === 'True' &&
                d.condition.type === 'Ready' ? (
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
                    Condition:
                    <b>
                      {d.condition === null
                        ? ' Initializing'
                        : d.condition.status !== 'True'
                        ? ' Waiting for: ' + d.condition.type
                        : ' ' + d.condition.type}
                    </b>
                  </span>
                  <span className="small">
                    Pod: <b>{d.pod_status}</b>
                  </span>
                  <span className="small">
                    Pod Name: <b>{d.pod_name}</b>
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

export default NotebookPage;
