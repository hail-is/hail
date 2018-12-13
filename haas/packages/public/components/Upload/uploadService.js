import Dropzone from 'dropzone';
import 'dropzone/dist/dropzone.css';
import { store } from 'react-easy-state';

// Because we rely on Auth, and on dropzone
// this file service m
import Auth from '../lib/Auth';

const uF = {};

let dropzone;

uF.state = store({
  files: [],
  dropzone: null
});

uF.initialize = () => {
  if (!uF.state.dropzone) {
    Dropzone = Dropzone || require('dropzone');
    Dropzone.autoDiscover = false;

    const updateOptions = uF.dropzoneSetup(dropzoneOptions);

    uF.state.dropzone = new Dropzone(document.body, updateOptions);

    uF.setupEvents();
  }

  return uF.state.dropzone;
};

uF.setupEvents = () => {
  uF.state.dropzone.on('addedfile', file => {
    if (!file) return;

    uF.state.files.push(file);
  });

  uF.state.dropzone.on('removedfile', file => {
    if (!file) return;

    const files = uF.state.files || [];

    files.forEach((fileInFiles, i) => {
      if (fileInFiles.name === file.name && fileInFiles.size === file.size) {
        files.splice(i, 1);
      }
    });

    console.info('after splice', uF.state.files);
  });
};

uF.dropzoneSetup = dropzoneOptions => {
  const opts = Object.assign({}, dropzoneOptions);
  // const previewNode = this.divTemplate;
  // console.info('div template', typeof this.divTemplate);
  // previewNode.id = '';

  // const previewTemplate = previewNode.parentNode.innerHTML;
  // // previewNode.parentNode.removeChild(previewNode);
  // console.info('previewTemplate', previewTemplate);
  // // dropzoneOptions.previewTemplate = previewTemplate;

  // const dropzoneNode = document.body; //ReactDOM.findDOMNode(this);

  // // Get the template HTML and remove it from the doumenthe template HTML and remove it from the doument
  // var previewNode = document.querySelector('#template');
  // previewNode.id = '';
  // var previewTemplate = previewNode.parentNode.innerHTML;
  // previewNode.parentNode.removeChild(previewNode);

  // opts.previewsContainer = '#previews';
  // opts.previewTemplate = previewTemplate;
  opts.clickable = '.dz-clickable';

  // It seems order of instantiation isn't from _app down?
  // This is called in _app, on componentDidMount,
  Auth.initialize();

  opts.headers = { Authorization: 'Bearer ' + Auth.state.idToken };

  console.info('auth token', Auth.state.idToken);

  return opts;
};
