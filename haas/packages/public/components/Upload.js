// An upload component
// Tutorials:
// https://www.dropzonejs.com/bootstrap.html#
//
import React from 'react';
import ReactDOM from 'react-dom';

import getConfig from 'next/config';

import Auth from 'lib/Auth';

// const styles = theme => ({
//   root: {
//     display: 'flex',
//     justifyContent: 'center',
//     alignItems: 'flex-end'
//   },
//   listContainer: {
//     width: '66vw',
//     boxShadow: 'none'
//   },
//   icon: {
//     margin: theme.spacing.unit * 2
//   },
//   iconHover: {
//     margin: theme.spacing.unit * 2,
//     '&:hover': {
//       color: red[800]
//     }
//   },
//   fileList: {
//     maxHeight: '70vh',
//     overflowY: 'scroll'
//   },
//   cardActions: {
//     alignItems: 'end'
//   },
//   fileItem: {
//     width: '100%',
//     marginRight: '15%',
//     wordBreak: 'break-word'
//   },
//   cancelButton: {
//     // position: 'absolute',
//     // left: '-4px',
//     // top: '-3px'
//   },
//   progressCircle: {
//     marginRight: 5
//     // top: '-18%',
//     // right: '50%',
//     // bottom: 0
//   }
// });

const { publicRuntimeConfig } = getConfig();

let Dropzone;
let dropzoneInstance;
const url = publicRuntimeConfig.API_UPLOAD_URL;

const dropzoneOptions = {
  url, // Set the url
  paramName: 'file', // The name that will be used to transfer the file
  maxFilesize: 1e6, // MB
  thumbnailWidth: 80,
  thumbnailHeight: 80,
  parallelUploads: 20,
  autoProcessQueue: false,
  previewsContainer: false,
  addRemoveLinks: false,
  timeout: 3e5 //about 83 hours
};

const snackbarAutohideDuration = 1000;

class SimpleSnackbar extends React.Component {
  state = {
    open: false
  };

  componentDidMount = () => {
    this.handleClick();
  };

  handleClick = () => {
    this.setState({ open: true });
  };

  handleClose = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }

    this.setState({ open: false });
  };

  render() {
    const { classes } = this.props;

    return (
      <div>
        <Snackbar
          anchorOrigin={{
            vertical: 'bottom',
            horizontal: 'center'
          }}
          open={this.state.open}
          autoHideDuration={snackbarAutohideDuration - 250}
          onClose={this.handleClose}
          ContentProps={{
            'aria-describedby': 'message-id'
          }}
          message={<span id="message-id">All Files Uploaded!</span>}
          action={[
            <IconButton
              key="close"
              aria-label="Close"
              color="inherit"
              className={classes.close}
              onClick={this.handleClose}
            >
              <CloseIcon />
            </IconButton>
          ]}
        />
      </div>
    );
  }
}

class Uploader extends React.Component {
  /**
   * React 'componentDidMount' method
   * Sets up dropzone.js with the component.
   */
  /**************************** Lifetime methods ******************************/
  constructor(props) {
    super(props);

    this.state = {
      files: this.dropzone ? this.dropzone.getActiveFiles() : [],
      success: false,
      u: false,
      showCancel: {}
    };
  }

  /**
   * React 'componentDidMount' : guaranteed to occur on client
   * Dropzone needs window, which is available after component mounts or updates
   **/
  componentDidMount = () => {
    this.dropzoneSingleton();
  };

  /**
   * React 'componentDidUpdate'
   * If the Dropzone hasn't been created, create it
   * Dropzone needs window, which is available after component mounts or updates
   **/
  componentDidUpdate = () => {
    this.queueDestroy = false;

    this.dropzoneSingleton();
  };

  /**
   * React 'componentWillUnmount'
   * Removes dropzone.js (and all its globals) if the component is being unmounted
   */
  componentWillUnmount() {
    if (this.dropzone) {
      // observe(() => {
      //   if(state.files.length == 0 && state.wantsToDestroy) {

      //   }
      // })

      const files = this.dropzone.getActiveFiles();

      if (files.length > 0) {
        // Well, seems like we still have stuff uploading.
        // This is dirty, but let's keep trying to get rid
        // of the dropzone until we're done here.
        this.queueDestroy = true;

        const destroyInterval = window.setInterval(() => {
          if (this.queueDestroy === false) {
            return window.clearInterval(destroyInterval);
          }

          if (this.dropzone.getActiveFiles().length === 0) {
            this.dropzone = this.destroy(this.dropzone);
            return window.clearInterval(destroyInterval);
          }
        }, 500);
      } else {
        this.dropzone = this.destroy(this.dropzone);
      }
    }
  }

  /**
   * Removes ALL listeners and Destroys dropzone. see https://github.com/enyo/dropzone/issues/1175
   */
  destroy() {
    this.dropzone.off();
    this.dropzone.destroy();
    dropzoneInstance = null;

    return null;
  }

  dropzoneSingleton = () => {
    if (!dropzoneInstance) {
      Dropzone = Dropzone || require('dropzone');
      Dropzone.autoDiscover = false;

      const updateOptions = this.dropzoneSetup(dropzoneOptions);

      dropzoneInstance = new Dropzone(document.body, updateOptions);

      this.dropzone = dropzoneInstance;

      this.setupEvents();
    }

    this.queueDestroy = false;
    return this.dropzone;
  };

  /***************************** Data methods *********************************/
  dropzoneSetup = dropzoneOptions => {
    const opts = Object.assign({}, dropzoneOptions);

    opts.clickable = '.dz-clickable';

    // It seems order of instantiation isn't from _app down?
    // This is called in _app, on componentDidMount,
    Auth.initialize();

    opts.headers = { Authorization: `Bearer ${Auth.state.idToken}` };

    return opts;
  };

  setupEvents() {
    this.dropzone.on('addedfile', file => {
      if (!file) return;

      const files = this.state.files || [];
      // this.dropzone.removeFile(file);
      files.push(file);
      this.setState({ files });
    });

    let timeout;
    this.dropzone.on('uploadprogress', (file, progress) => {
      if (timeout) {
        clearTimeout(timeout);
      }

      timeout = setTimeout(() => {
        this.forceUpdate();
      }, 33);
    });

    this.dropzone.on('removedfile', file => {
      if (!file) return;

      if (!this.state.files) {
        return;
      }

      const files = this.state.files.filter(eFile => {
        return eFile.upload.uuid !== file.upload.uuid;
      });

      this.setState({ files });
    });

    this.dropzone.on('queuecomplete', progress => {
      // if (progress === 100) {
      this.dropzone.removeAllFiles(true);
      this.setState({ files: [], uploading: false, success: true });

      // A duration slighlty higher than autohide
      setTimeout(() => {
        this.setState({ success: false });
        this.forceUpdate();
      }, snackbarAutohideDuration + 50);
    });
  }

  deleteFileFromList = (file, index) => {
    this.dropzone.removeFile(file);
    this.state.files.splice(index, 1);
    this.setState({ files: this.state.files });
  };

  handleUploadStart = () => {
    this.dropzone.processQueue();
    this.setState({ uploading: true });
  };

  showOrHide = index => {
    this.setState(state => {
      const showCancel = state.showCancel || {};

      showCancel[index] = !showCancel[index];

      return {
        showCancel: showCancel
      };
    });
  };

  render() {
    const { classes } = this.props;

    const { uploading, success } = this.state;

    // const buttonClassname = classNames({
    //   [classes.buttonSuccess]: success
    // });

    return (
      <React.Fragment>
        <div
          style={{
            display:
              this.state.files.length == 0 && !this.state.success
                ? 'inherit'
                : 'none'
          }}
          className={`dz-clickable`}
          aria-label="Upload"
        >
          Click
        </div>
      </React.Fragment>
    );
  }
}

export default Uploader;
