import auth from '../../libs/auth';
import fetch from 'isomorphic-unfetch';
import getConfig from 'next/config';

// TODO: Improve separation
declare type runningStatus = {
  started_at: number;
};

declare type waitingStatus = {
  reason: string;
};

declare type terminatedStatus = {
  exit_code: number;
  finished_at: number;
  started_at: number;
  reason: string;
};

declare type kubeWebsocketUpdate = {
  event: kubeEvent;
  resource: notebookType;
};

enum kubeEvent {
  ADDED = 'ADDDED',
  DELETED = 'DELETED',
  MODIFIED = 'MODIFIED'
}

export type notebookType = {
  svc_name: string;
  name: string;
  pod_name: string;
  token: string;
  creation_date: string;
  svc_status: string;
  pod_status: string; // The kubernetes status.phase val for pod
  container_status: {
    running: runningStatus | null;
    waiting: waitingStatus | null;
    terminated: terminatedStatus | null;
  } | null;
  condition: {
    message: string | null;
    reason: string | null;
    status: string | null;
    type: string | null;
  } | null;
};

export type notebooksType = { [pod_name: string]: notebookType };

export enum kubeState {
  Deleted = 'Deleted',
  Running = 'Running'
}

// TODO: If user isn't logged in, and make it to this page
// TODO: Enumerate waitingStatus 'reason's
// after 401, tell them they need to log in
const cfg = getConfig().publicRuntimeConfig.NOTEBOOK;
export const URL: string = cfg.URL;

const getAlive = (nbs: notebooksType) =>
  Object.values(nbs).filter(
    (d: notebookType) => d.svc_status != kubeState.Deleted
  );

let resultsPromise: Promise<{
  notebooks: notebooksType;
  alive: notebookType[];
}>;

let notebooks: notebooksType = {};
let alive: notebookType[] = [];

let initialized = false;
let isListening: boolean;

export const Notebook = {
  get notebooks() {
    return notebooks;
  },
  get aliveNotebooks() {
    return alive;
  },
  get initialized() {
    return initialized;
  }
};

export const startRequest = () => {
  if (resultsPromise) {
    return resultsPromise;
  }

  resultsPromise = fetch(`${URL}/api`, {
    headers: {
      Authorization: `Bearer ${auth.accessToken}`
    }
  })
    .then(d => d.json())
    .then((aNotebooks: notebookType[]) => {
      const data: notebooksType = {};

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

export const callbackFunctions: ((
  nb: notebooksType,
  alive: notebookType[]
) => void)[] = [];

export const startListener = () => {
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

export const createNotebook = async () => {
  return fetch(`${URL}/api`, {
    headers: {
      Authorization: `Bearer ${auth.accessToken}`
    },
    method: 'POST'
  })
    .then(d => d.json())
    .then((notebook: notebookType) => {
      notebooks[notebook.pod_name] = notebook;
      alive = getAlive(notebooks);

      return { notebooks, alive };
    });
};

export const removeItemFromNotebook = (notebook: notebookType) => {
  delete notebooks[notebook.pod_name];
  alive = getAlive(notebooks);
};

export const deleteNotebook = (notebook: notebookType, skipRemove: boolean) => {
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
