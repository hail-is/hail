import React from "react";
import PropTypes from "prop-types";
import deburr from "lodash/deburr";
import Downshift from "downshift";
import { withStyles } from "@material-ui/core/styles";
import TextField from "@material-ui/core/TextField";
import Paper from "@material-ui/core/Paper";
import MenuItem from "@material-ui/core/MenuItem";
import Grid from "@material-ui/core/Grid";
import { Query } from "react-apollo";
import ErrorMessage from "../ErrorMessage";
import CircularProgress from "@material-ui/core/CircularProgress";

import gql from "graphql-tag";
import Typography from "@material-ui/core/Typography";

function renderInput(inputProps) {
  const { InputProps, classes, ref, ...other } = inputProps;

  return (
    <TextField
      InputProps={{
        inputRef: ref,
        classes: {
          root: classes.inputRoot,
          input: classes.inputInput
        },
        ...InputProps
      }}
      {...other}
    />
  );
}

function renderSuggestion({
  job,
  idx,
  itemProps,
  highlightedIndex,
  selectedItem
}) {
  // const isHighlighted = highlightedIndex === idx;
  const isSelected = (selectedItem || "").indexOf(job.name) > -1;

  const date = new Date(Number(job.createdAt));

  return (
    <MenuItem
      alignItems="flex-start"
      {...itemProps}
      key={idx}
      // selected={isHighlighted}
      component="div"
      style={{
        flexDirection: "column",
        minHeight: 48,
        overflowX: "scroll"
      }}
    >
      <div>
        <span style={{ fontWeight: isSelected ? 700 : 400 }}>{job.name}</span>
        <span style={{ display: "block" }}>
          Created on: {date.toDateString()}
        </span>
      </div>
    </MenuItem>
    // </MenuItem>
  );
}
renderSuggestion.propTypes = {
  highlightedIndex: PropTypes.number,
  idx: PropTypes.number,
  itemProps: PropTypes.object,
  selectedItem: PropTypes.string,
  job: PropTypes.shape({ name: PropTypes.string, createdAt: PropTypes.number })
    .isRequired
};

function getSuggestions(allJobs, value) {
  const inputValue = deburr(value.trim()).toLowerCase();
  const inputLength = inputValue.length;
  let count = 0;

  return inputLength === 0
    ? allJobs
    : allJobs.filter(job => {
        const keep =
          count < 5 &&
          job.name.slice(0, inputLength).toLowerCase() === inputValue;

        if (keep) {
          count += 1;
        }

        return keep;
      });
}

const styles = theme => ({
  root: {
    flexGrow: 1,
    height: 250
  },
  container: {
    flexGrow: 1,
    position: "relative"
  },
  paper: {
    position: "absolute",
    zIndex: 1,
    marginTop: theme.spacing.unit,
    left: 0,
    right: 0,
    maxHeight: 300,
    overflowY: "scroll"
  },
  chip: {
    margin: `${theme.spacing.unit / 2}px ${theme.spacing.unit / 4}px`
  },
  inputRoot: {
    flexWrap: "wrap"
  },
  inputInput: {
    width: "auto",
    flexGrow: 1
  },
  divider: {
    height: theme.spacing.unit * 2
  }
});

const GET_JOBS = gql`
  {
    jobs {
      id
      createdAt
      name
      assembly
      updatedAt
      submission {
        state
        attempts
      }
    }
  }
`;

const ChooseJobs = ({ classes, onSelected }) => {
  return (
    <Query query={GET_JOBS}>
      {({ loading, error, data, fetchMore }) => {
        if (error) {
          return <ErrorMessage error={error} />;
        }

        if (loading) return <CircularProgress />;

        return (
          <Grid
            container
            spacing={24}
            direction="row"
            justify="space-evenly"
            alignItems="flex-start"
            style={{
              width: "70vw",
              overflow: "visible",
              textAlign: "left"
            }}
          >
            <Grid item xs={12}>
              <Typography variant="h4" component="h4">
                View jobs
              </Typography>
              <Typography variant="subtitle2" gutterBottom>
                Check an existing analysis
              </Typography>
            </Grid>
            <Grid item xs={12}>
              <Downshift id="downshift-simple">
                {({
                  getInputProps,
                  getItemProps,
                  getMenuProps,
                  highlightedIndex,
                  inputValue,
                  isOpen,
                  selectedItem,
                  openMenu
                }) => (
                  <div className={classes.container}>
                    {renderInput({
                      fullWidth: true,
                      classes,
                      InputProps: getInputProps({
                        placeholder: "Search for a job by name",
                        onClick: () => openMenu()
                      })
                    })}
                    <div {...getMenuProps()}>
                      {isOpen ? (
                        <Paper className={classes.paper} square>
                          {getSuggestions(data.jobs, inputValue).map(
                            (job, idx) =>
                              renderSuggestion({
                                job,
                                idx,
                                itemProps: getItemProps({
                                  item: job.name,
                                  full: job,
                                  onClick: () => onSelected(job, idx)
                                }),
                                highlightedIndex,
                                selectedItem
                              })
                          )}
                        </Paper>
                      ) : null}
                    </div>
                  </div>
                )}
              </Downshift>
            </Grid>
          </Grid>
        );
      }}
    </Query>
  );
};

ChooseJobs.propTypes = {
  classes: PropTypes.object.isRequired,
  onSelected: PropTypes.func.isRequired
};

export default withStyles(styles)(ChooseJobs);
