// Example component that allows one to select samples from fam file
import React from 'react';
import PropTypes from 'prop-types';

import { Query } from 'react-apollo';
import gql from 'graphql-tag';

import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import CardActions from '@material-ui/core/CardActions';
import Grid from '@material-ui/core/Grid';

import Typography from '@material-ui/core/Typography';
import Button from '@material-ui/core/Button';

import { AgGridReact } from 'ag-grid-react';

// For some reason, there must be at least 1 import css/less statement
// in _app.js (or maybe _document?), in order that these css files
// get imported correctly; without that, the route in which this
// component is requested will never load
// import 'ag-grid-community/dist/styles/ag-grid.css';
// import 'ag-grid-community/dist/styles/ag-theme-material.css';

import ErrorMessage from '../ErrorMessage';
import CircularProgress from '@material-ui/core/CircularProgress';

const famHeader = ['FID', 'IID', 'FatherID', 'MotherID', 'Sex', 'Phenotype'];

const GET_FILE_CONTENTS = gql`
  query s3File($url: String!) {
    s3File(url: $url) {
      contents
    }
  }
`;

const agColumnDefs = famHeader.map((val, idx) => {
  const obj = {};

  if (idx == 0) {
    obj.checkboxSelection = true;
  }

  obj.headerName = val;
  obj.field = val;

  return obj;
});

const parseFamForAg = str => {
  // const result = [];

  return str
    .trim()
    .split('\n')
    .map(val => {
      const row = {};
      val.split(/\s/).forEach((field, idx) => {
        // const obj = {};
        row[famHeader[idx]] = field;
      });

      return row;
    });

  // return result;
};

class SampleGrid extends React.Component {
  constructor(props) {
    super(props);

    const famFile = props.famFile;

    this.state = {
      selectedSamples: [],
      renderedFam: null
    };
  }

  // TODO: memoize
  processFile = str => {
    if (this.state.renderedFam) {
      return this.state.renderedFam;
    }

    // if (type === 'fam') {
    const rows = parseFamForAg(str);

    this.setState({
      renderedFam: rows
    });

    return rows;
  };

  sizeToFit() {
    this.gridApi.sizeColumnsToFit();
  }

  autoSizeAll() {
    var allColumnIds = [];
    this.gridColumnApi.getAllColumns().forEach(function(column) {
      allColumnIds.push(column.colId);
    });
    this.gridColumnApi.autoSizeColumns(allColumnIds);
  }
  onGridReady(params) {
    this.gridApi = params.api;
    this.gridColumnApi = params.columnApi;
    this.autoSizeAll();
  }

  handleRowSelected = () => {
    const rows = this.gridApi.getSelectedRows();
    this.setState({ selectedSamples: rows });
  };

  clearFiles = () => {
    this.setState({
      selectedSamples: []
    });
  };

  render() {
    return (
      <Query
        query={GET_FILE_CONTENTS}
        variables={{ url: this.props.famFilePath }}
        onCompleted={data => this.processFile(data.s3File.contents)}
        notifyOnNetworkStatusChange
      >
        {({ loading, error, data, fetchMore }) => {
          if (error) return <ErrorMessage error={error} />;
          if (loading) return <CircularProgress />;

          return (
            <Card>
              <style jsx>{`
                :global(.heading) {
                  text-align: left;
                }

                .ag-theme-material {
                  height: 50vh;
                  width: 70vw;
                  text-align: left;
                  max-width: 840px;
                }
                .ag-theme-material .ag-filter-select {
                  margin: 8px;
                  width: calc(100% - 16px);
                  padding: 5px;
                  margin: 8px;
                  width: calc(100% - 16px);
                  padding: 5px;
                }

                .ag-theme-material .ag-root-wrapper.ag-layout-normal {
                  height: 95% !important;
                }
              `}</style>
              <CardContent>
                <Typography variant="h4" className="heading">
                  Select Samples
                </Typography>
                <Typography variant="subtitle2" className="heading">
                  {this.state.selectedSamples.length} selected
                </Typography>

                <div className="ag-theme-material">
                  <AgGridReact
                    enableSorting={true}
                    enableFilter={true}
                    rowSelection="multiple"
                    columnDefs={agColumnDefs}
                    enableColResize={true}
                    rowData={this.state.renderedFam}
                    onGridReady={this.onGridReady.bind(this)}
                    onSelectionChanged={this.handleRowSelected}
                  />
                </div>
              </CardContent>
              <CardActions>
                <Button color="primary" onClick={this.props.onBackClicked}>
                  Back
                </Button>
                <Button
                  color="primary"
                  onClick={() =>
                    this.props.onSamplesSelected(
                      this.state.selectedSamples,
                      this.state.selectedFiles
                    )
                  }
                >
                  Next
                </Button>
              </CardActions>
            </Card>
          );
        }}
      </Query>
    );
  }
}

SampleGrid.propTypes = {
  famFilePath: PropTypes.string.isRequired,
  onSamplesSelected: PropTypes.func.isRequired,
  onBackClicked: PropTypes.func.isRequired
};

export default SampleGrid;
