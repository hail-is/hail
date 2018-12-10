import React from 'react';
import PropTypes from 'prop-types';

import { Query } from 'react-apollo';
import gql from 'graphql-tag';

import { withStyles } from '@material-ui/core/styles';
import MenuItem from '@material-ui/core/MenuItem';

import Select from 'react-select';
import ErrorMessage from '../ErrorMessage';

import Typography from '@material-ui/core/Typography';
import NoSsr from '@material-ui/core/NoSsr';
import TextField from '@material-ui/core/TextField';
import Paper from '@material-ui/core/Paper';

import CircularProgress from '@material-ui/core/CircularProgress';

import MaterialGrid from '@material-ui/core/Grid';

const famRegex = new RegExp(/\.fam$|\.fam.gz$/);

const GET_S3_OBJECTS = gql`
  {
    allS3Objects {
      lastModified
      name
      ownerName
    }
  }
`;

function NoOptionsMessage(props) {
  return (
    <Typography
      color="textSecondary"
      className={props.selectProps.classes.noOptionsMessage}
      {...props.innerProps}
    >
      {props.children}
    </Typography>
  );
}

function inputComponent({ inputRef, ...props }) {
  return <div ref={inputRef} {...props} />;
}

function Control(props) {
  return (
    <TextField
      fullWidth
      InputProps={{
        inputComponent,
        inputProps: {
          className: props.selectProps.classes.input,
          inputRef: props.innerRef,
          children: props.children,
          ...props.innerProps
        }
      }}
      {...props.selectProps.textFieldProps}
    />
  );
}

function Option(props) {
  return (
    <MenuItem
      buttonRef={props.innerRef}
      selected={props.isFocused}
      component="div"
      style={{
        fontWeight: props.isSelected ? 500 : 400
      }}
      {...props.innerProps}
    >
      {props.children}
    </MenuItem>
  );
}

function Placeholder(props) {
  return (
    <Typography
      color="textSecondary"
      className={props.selectProps.classes.placeholder}
      {...props.innerProps}
    >
      {props.children}
    </Typography>
  );
}

function SingleValue(props) {
  return (
    <Typography
      className={props.selectProps.classes.singleValue}
      {...props.innerProps}
    >
      {props.children}
    </Typography>
  );
}

function ValueContainer(props) {
  return (
    <div className={props.selectProps.classes.valueContainer}>
      {props.children}
    </div>
  );
}

function SelectMenu2(props) {
  return (
    <Paper className={props.selectProps.classes.paper} {...props.innerProps}>
      {props.children}
    </Paper>
  );
}

const components = {
  Control,
  Menu: SelectMenu2,
  NoOptionsMessage,
  Option,
  Placeholder,
  SingleValue,
  ValueContainer
};

const styles = theme => ({
  input: {
    display: 'flex',
    padding: 0
  },
  valueContainer: {
    display: 'flex',
    // flexWrap: 'wrap',
    whiteSpace: 'nowrap',
    textOverflow: 'ellipsis',
    flex: 1,
    alignItems: 'center',
    overflow: 'hidden'
  },
  noOptionsMessage: {
    padding: `${theme.spacing.unit}px ${theme.spacing.unit * 2}px`
  },
  singleValue: {
    // fontSize: 16,
    overflow: 'hidden',
    textOverflow: 'ellipsis'
  },
  placeholder: {
    position: 'absolute',
    left: 2,
    fontSize: 16
  },
  paper: {
    position: 'absolute',
    zIndex: 1,
    marginTop: theme.spacing.unit,
    left: 0,
    right: 0
  }
});

class ChooseData extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      selectedFiles: {
        fam: '',
        data: ''
      }
    };
  }

  clearFiles = () => {
    this.setState({
      selectedFiles: {
        fam: '',
        data: ''
      }
    });
  };

  handleSelect = (type, callbackIfDone) => selected => {
    // Unfortunately, setState is in no way synchronous,
    // so we need to internally mutate it, or clone it, whatever
    // if we want to check within this function
    this.state.selectedFiles[type] = selected;
    this.setState({
      selectedFiles: this.state.selectedFiles
    });

    const famFile = this.state.selectedFiles.fam;
    const dataFile = this.state.selectedFiles.data;

    if (famFile && dataFile) {
      callbackIfDone(this.state.selectedFiles);
      return;
    }
  };

  render() {
    const { classes, onFilesSelected } = this.props;

    return (
      <Query query={GET_S3_OBJECTS}>
        {({ loading, error, data, fetchMore }) => {
          if (error) return <ErrorMessage error={error} />;
          if (loading) return <CircularProgress />;

          return (
            <NoSsr fallback={<CircularProgress />}>
              <MaterialGrid
                container
                spacing={24}
                direction="row"
                justify="space-evenly"
                alignItems="flex-start"
                style={{
                  width: '70vw',
                  overflow: 'visible',
                  textAlign: 'left'
                }}
              >
                {/* <div className={classes.root}> */}
                <MaterialGrid item xs={12}>
                  <Typography variant="h4" component="h4">
                    File selection
                  </Typography>
                  <Typography variant="subtitle2" gutterBottom>
                    Grab some files for analysis
                  </Typography>
                </MaterialGrid>
                <MaterialGrid item xs={12} sm={6}>
                  <Select
                    classes={classes}
                    // styles={selectStyles}
                    name="Data select"
                    // components={{ Menu: SelectMenu }}
                    options={data.allS3Objects
                      .filter(item => item.name.match(famRegex) === null)
                      .map(item => {
                        return {
                          label: item.name,
                          value: item.name,
                          obj: item
                        };
                      })}
                    components={components}
                    value={this.state.selectedFiles.data}
                    onChange={this.handleSelect('data', onFilesSelected)}
                    placeholder="Select a data file"
                  />
                </MaterialGrid>
                <MaterialGrid item xs={12} sm={6}>
                  <Select
                    classes={classes}
                    // styles={selectStyles}
                    name="famSelect"
                    placeholder={`Select a .fam
                           file`}
                    options={data.allS3Objects
                      .filter(item => item.name.match(famRegex))
                      .map(item => {
                        return {
                          value: item.name,
                          label: item.name,
                          obj: item
                        };
                      })}
                    components={components}
                    value={this.state.selectedFiles.fam}
                    onChange={this.handleSelect('fam', onFilesSelected)}
                  />
                </MaterialGrid>
              </MaterialGrid>
            </NoSsr>
          );
        }}
      </Query>
    );
  }
}

ChooseData.propTypes = {
  classes: PropTypes.object.isRequired,
  onFilesSelected: PropTypes.func.isRequired
};

export default withStyles(styles)(ChooseData);
