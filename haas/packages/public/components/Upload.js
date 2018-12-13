// An upload component
// Tutorials:
// https://www.dropzonejs.com/bootstrap.html#
//
import React from "react";
import ReactDOM from "react-dom";

import getConfig from "next/config";

// import 'dropzone/dist/dropzone.css';

import PropTypes from "prop-types";
import { withStyles } from "@material-ui/core/styles";
import red from "@material-ui/core/colors/red";
import IconButton from "@material-ui/core/IconButton";
import Button from "@material-ui/core/Button";

import List from "@material-ui/core/List";
import ListItem from "@material-ui/core/ListItem";
import ListItemSecondaryAction from "@material-ui/core/ListItemSecondaryAction";
import ListItemText from "@material-ui/core/ListItemText";
import CircularProgress from "@material-ui/core/CircularProgress";
import Card from "@material-ui/core/Card";
import CardActions from "@material-ui/core/CardActions";
import HighlightOffRounded from "@material-ui/icons/HighlightOffRounded";
import { store, view } from "react-easy-state";
import classNames from "classnames";
import Grid from "@material-ui/core/Grid";
import CloudUploadRounded from "@material-ui/icons/CloudUploadRounded";
import Tooltip from "@material-ui/core/Tooltip";
import Snackbar from "@material-ui/core/Snackbar";
import CloseIcon from "@material-ui/icons/Close";
import DoneAll from "@material-ui/icons/DoneAllRounded";

import Auth from "../lib/Auth";

const styles = theme => ({
  root: {
    display: "flex",
    justifyContent: "center",
    alignItems: "flex-end"
  },
  listContainer: {
    width: "66vw",
    boxShadow: "none"
  },
  icon: {
    margin: theme.spacing.unit * 2
  },
  iconHover: {
    margin: theme.spacing.unit * 2,
    "&:hover": {
      color: red[800]
    }
  },
  fileList: {
    maxHeight: "70vh",
    overflowY: "scroll"
  },
  cardActions: {
    alignItems: "end"
  },
  fileItem: {
    width: "100%",
    marginRight: "15%",
    wordBreak: "break-word"
  },
  cancelButton: {
    // position: 'absolute',
    // left: '-4px',
    // top: '-3px'
  },
  progressCircle: {
    marginRight: 5
    // top: '-18%',
    // right: '50%',
    // bottom: 0
  }
});

const { publicRuntimeConfig } = getConfig();

let Dropzone;
let dropzoneInstance;
const url = publicRuntimeConfig.API_UPLOAD_URL;

const dropzoneOptions = {
  url, // Set the url
  paramName: "file", // The name that will be used to transfer the file
  maxFilesize: 1e6, // MB
  thumbnailWidth: 80,
  thumbnailHeight: 80,
  parallelUploads: 20,
  autoProcessQueue: false,
  previewsContainer: false,
  addRemoveLinks: false,
  timeout: 3e5 //about 83 hours
};

// const state = store({
//   files: [],
// });

const state = store({
  files: [],
  success: false,
  uploading: false
});

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
    if (reason === "clickaway") {
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
            vertical: "bottom",
            horizontal: "center"
          }}
          open={this.state.open}
          autoHideDuration={snackbarAutohideDuration - 250}
          onClose={this.handleClose}
          ContentProps={{
            "aria-describedby": "message-id"
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

SimpleSnackbar.propTypes = {
  classes: PropTypes.object.isRequired
};

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
      Dropzone = Dropzone || require("dropzone");
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

    opts.clickable = ".dz-clickable";

    // It seems order of instantiation isn't from _app down?
    // This is called in _app, on componentDidMount,
    Auth.initialize();

    opts.headers = { Authorization: `Bearer ${Auth.state.idToken}` };

    return opts;
  };

  setupEvents() {
    this.dropzone.on("addedfile", file => {
      if (!file) return;

      const files = this.state.files || [];
      // this.dropzone.removeFile(file);
      files.push(file);
      this.setState({ files });
    });

    let timeout;
    this.dropzone.on("uploadprogress", (file, progress) => {
      if (timeout) {
        clearTimeout(timeout);
      }

      timeout = setTimeout(() => {
        this.forceUpdate();
      }, 33);
    });

    this.dropzone.on("removedfile", file => {
      if (!file) return;

      if (!this.state.files) {
        return;
      }

      const files = this.state.files.filter(eFile => {
        return eFile.upload.uuid !== file.upload.uuid;
      });

      this.setState({ files });
    });

    this.dropzone.on("queuecomplete", progress => {
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

    const buttonClassname = classNames({
      [classes.buttonSuccess]: success
    });

    return (
      <React.Fragment>
        {/* {this.state.files.length == 0 && ( */}
        {/* Can't remove from DOM, because dropzone won't know to re-initialize */}
        <Tooltip
          title="Upload (Click or Drop)"
          style={{
            display:
              this.state.files.length == 0 && !this.state.success
                ? "inline"
                : "none"
          }}
        >
          <IconButton
            style={{
              display:
                this.state.files.length == 0 && !this.state.success
                  ? "inherit"
                  : "none"
            }}
            className={`${classes.button} dz-clickable`}
            aria-label="Upload"
          >
            <CloudUploadRounded style={{ fontSize: 48 }} />
            {/* <GetAppIcon /> */}
          </IconButton>
        </Tooltip>
        <DoneAll
          style={{
            display: this.state.success ? "inline" : "none",
            fontSize: 48
          }}
        />
        {this.state.success ? <SimpleSnackbar classes={classes} /> : ""}
        <Card className={classes.listContainer}>
          <List className={classes.fileList} id="previews">
            <style jsx>{`
              .hover-button .hover-button--on,
              .hover-button:hover .hover-button--off {
                display: none;
              }

              .hover-button:hover .hover-button--on {
                display: inline;
              }
            `}</style>
            {this.state.files.map((file, index) => (
              <ListItem key={file.upload.uuid}>
                <ListItemText
                  primary={file.name}
                  secondary={`${
                    file.size
                  } bytes (${file.upload.progress.toFixed(2)}\% uploaded)`}
                  className={classes.fileItem}
                />
                <ListItemSecondaryAction className="hover-button">
                  {/* <button data-dz-remove className="btn btn-danger delete">
                      <i className="glyphicon glyphicon-trash" />
                      <span>Delete</span>
                    </button> */}
                  <React.Fragment>
                    {file.status === "uploading" ||
                    file.status === "success" ? (
                      file.upload.progress < 100 ? (
                        <React.Fragment>
                          {file.upload.progress < 5 ? (
                            <CircularProgress className="hover-button--off" />
                          ) : (
                            <CircularProgress
                              className={`${
                                classes.progressCircle
                              } hover-button--off`}
                              variant="determinate"
                              value={file.upload.progress}
                            />
                          )}

                          <IconButton
                            className={`${
                              classes.cancelButton
                            }  hover-button--on`}
                            onClick={() => this.dropzone.removeFile(file)}
                          >
                            <HighlightOffRounded />
                          </IconButton>
                        </React.Fragment>
                      ) : (
                        <div>Done</div>
                      )
                    ) : (
                      <IconButton
                        onClick={() => this.dropzone.removeFile(file)}
                      >
                        <HighlightOffRounded />
                      </IconButton>
                    )}
                    {/* <Button onClick={() => this.dropzone.processFile(file)}>
                    Start
                  </Button> */}
                  </React.Fragment>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
          {this.state.files.length && !this.state.uploading ? (
            <CardActions className={classes.cardActions}>
              <Grid
                container
                alignItems="center"
                direction="column"
                justify="center"
              >
                <Grid item>
                  <Button onClick={this.handleUploadStart}>Start</Button>
                </Grid>
              </Grid>
            </CardActions>
          ) : (
            ""
          )}
        </Card>
      </React.Fragment>
    );
  }
}

Uploader.propTypes = {
  classes: PropTypes.object.isRequired
  // backAction: PropTypes.
};

export default withStyles(styles)(view(Uploader));
