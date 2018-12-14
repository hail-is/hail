import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import FormControl from '@material-ui/core/FormControl';
import Select from '@material-ui/core/Select';
import gql from 'graphql-tag';
import { Mutation } from 'react-apollo';
import { view } from 'react-easy-state';
import fetch from 'isomorphic-unfetch';
import Upload from '../components/Upload';
import Auth from '../lib/Auth';

const styles = theme => ({
  root: {
    display: 'flex',
    flexWrap: 'wrap',
    flexDirection: 'column'
  },
  formControl: {
    margin: theme.spacing.unit,
    minWidth: 150
  },
  selectEmpty: {
    marginTop: theme.spacing.unit * 2
  },
  card: {
    minWidth: 275
  },
  bullet: {
    display: 'inline-block',
    margin: '0 2px',
    transform: 'scale(0.8)'
  },
  title: {
    fontSize: 14
  },
  pos: {
    marginBottom: 12
  }
});

const species = ['Human', 'Mouse', 'Rhesus', 'Rat'];
const genomes = {
  Human: [
    {
      id: 1,
      name: 'GRCh38/hg38',
      date: 2013,
      value: 'hg38'
    },
    {
      id: 1,
      name: 'GRCh37/hg19',
      date: 2009,
      value: 'hg19'
    }
  ],
  Mouse: [
    {
      id: 1,
      name: 'GRCm38/mm10',
      date: 2011,
      value: 'mm10'
    },
    {
      id: 2,
      name: 'NCBI37/mm9',
      date: 2007,
      value: 'mm9'
    }
  ],
  Rhesus: [
    {
      id: 1,
      name: 'BCM Mmul_8.0.1/rheMac8',
      date: 2013,
      value: 'rheMac8'
    }
  ],
  Rat: [
    {
      id: 1,
      name: 'RGSC 6.0/rn6',
      date: 2014,
      value: 'rn6'
    }
  ],
  'D. melanogaster': [
    {
      id: 1,
      name: 'GRCm38/mm10',
      date: 2011,
      value: 'mm10'
    },
    {
      id: 2,
      name: 'NCBI37/mm9',
      date: 2007,
      value: 'mm9'
    }
  ]
};

const state = {
  submittedSpecies: false,
  setAssembly: false,
  setEmail: false,
  setFile: false,
  setOptions: false
};

function SelectAssembly(props) {
  const { organism, assembly, classes, fn } = props;

  const assemblies = genomes[organism];

  return (
    <React.Fragment>
      <FormControl className={classes}>
        <InputLabel htmlFor="assembly-label-placeholder">Assembly</InputLabel>
        <Select value={assembly.name} onChange={fn}>
          <MenuItem value="">
            <em>None</em>
          </MenuItem>
          {assemblies &&
            assemblies.map(ass => (
              <MenuItem value={ass.name} key={ass.value}>
                <em>{ass.name}</em>
              </MenuItem>
            ))}
        </Select>
      </FormControl>
    </React.Fragment>
  );
}

function SelectSpecies(props) {
  const { organism, classes, fn } = props;

  return (
    <React.Fragment>
      <FormControl className={classes}>
        <InputLabel htmlFor="species-label-placeholder">Species</InputLabel>
        <Select value={organism} onChange={fn}>
          <MenuItem value="">
            <em>None</em>
          </MenuItem>
          {species.map(organism => (
            <MenuItem value={organism} key={organism}>
              <em>{organism}</em>
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    </React.Fragment>
  );
}

class Submit extends React.Component {
  state = {
    organism: '',
    assembly: { name: '' },
    email: '',
    speciesAssemblies: [],
    stuff: '',
    setSpecies: false,
    setAssembly: false,
    setEmail: false,
    setFile: false,
    setOptions: false
  };

  componentDidMount = async () => {
    try {
      const val = fetch('http://localhost:8000/api/ci', {
        headers: {
          Authorization: `Bearer ${Auth.getAccessToken()}`
        }
      });
      console.info(val);
    } catch (e) {
      console.error(e);
    }
  };

  updateSpecies = event => {
    const species = event.target.value;

    let setSpecies = false;
    if (species) {
      setSpecies = true;
    }
    this.setState({
      organism: event.target.value,
      speciesAssemblies: genomes[event.target.value] || [],
      assembly: { name: '' },
      stuff: '',
      setSpecies
    });
  };

  updateAssembly = event => {
    const assembly = event.target.value;

    let setAssembly = false;
    if (assembly) {
      setAssembly = true;
    }

    this.setState({
      assembly: { name: event.target.value },
      setAssembly
    });

    console.info(this.state);
  };

  render() {
    const { classes } = this.props;

    return <Upload />;
  }
}

Submit.propTypes = {
  classes: PropTypes.object.isRequired
};

export default withStyles(styles)(view(Submit));
