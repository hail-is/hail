import { PureComponent, Fragment } from 'react';
import dynamic from 'next/dynamic';

const MonacoEditor = dynamic(() => import('components/Editor'), {
  ssr: false
});

class App extends PureComponent {
  constructor(props) {
    super(props);
    this.state = {
      code: ''
    };
  }
  editorDidMount = (editor, monaco) => {
    console.log('editorDidMount', editor);
    editor.focus();
  };
  onChange = (newValue, e) => {
    console.log('onChange', newValue, e);
    this.setState({
      code: newValue
    });
  };
  render() {
    const code = this.state.code;

    return (
      <span
        style={{
          display: 'flex',
          flexDirection: 'row'
        }}
      >
        <MonacoEditor
          width="67vw"
          height="90vh"
          language="python"
          theme="vs-dark"
          value={code}
          onChange={this.onChange}
          editorDidMount={this.editorDidMount}
        />
        <div style={{ width: '33vw', padding: 15 }}>{this.state.code}</div>
      </span>
    );

    // return <div>Loading</div>;
  }
}

export default App;
